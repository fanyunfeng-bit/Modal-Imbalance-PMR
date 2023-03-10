import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier, CLClassifier, AClassifier, VClassifier
# from models.contrastive_model import AVConEncoder, AVConClassifier
from utils.utils import setup_seed, weight_init
import torch.nn.functional as F


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--fps', default=1, type=int, help='Extract how many frames in a second')
    parser.add_argument('--num_frame', default=1, type=int, help='use how many frames for train')

    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')
    parser.add_argument('--load_path', default='ckpt/Method-CE/model-CREMAD-concat-bsz128-embed_dim-512', type=str,
                        help='path to load trained model')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    return parser.parse_args()


def dot_product_angle_tensor(v1, v2):
    vector_dot_product = torch.dot(v1, v2)
    arccos = torch.acos(vector_dot_product / (torch.norm(v1, p=2) * torch.norm(v2, p=2)))
    angle = np.degrees(arccos.data.cpu().numpy())
    return arccos, angle


def train_uniclassifier_epoch(args, epoch, model, device, dataloader, optimizer,
                              scheduler):
    criterion = nn.CrossEntropyLoss()

    model.audio_net.eval()
    model.visual_net.eval()
    model.fusion_module.train()
    # encoder.eval()
    print("Start training classifier ... ")

    _loss = 0
    # _a_angle = 0
    # _v_angle = 0
    #
    # angle_file = args.logs_path + '/combine' + '/angle-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
    #              str(args.batch_size) + '-lr' + str(args.learning_rate) + '.txt'
    # if not os.path.exists(args.logs_path + '/combine'):
    #     os.makedirs(args.logs_path + '/combine')
    #
    # if (os.path.isfile(angle_file)):
    #     os.remove(angle_file)  # 删掉已有同名文件
    # f_angle = open(angle_file, 'a')

    for step, (spec, image, label) in enumerate(dataloader):
        spec = spec.to(device)  # B x 257 x 1004
        image = image.to(device)  # B x 3(image count) x 3 x 224 x 224
        label = label.to(device)  # B
        # B = label.shape[0]

        optimizer.zero_grad()

        with torch.no_grad():
            a = model.audio_net(spec.unsqueeze(1).float())

            v = model.visual_net(image.float())
            
        for name, params in model.named_parameters():
            layer = str(name).split('.')[0]

            print(name, layer, params.size())

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = model.fusion_module(a, v)

        loss = criterion(out, label)

        print('loss: ', loss)

        loss.backward()
        optimizer.step()

        _loss += loss.item()
    scheduler.step()
    return _loss / len(dataloader)


def valid_uniclassifier(args, model, device, dataloader):
    model.eval()

    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)
            B = label.shape[0]
            a, v, out = model(spec.unsqueeze(1).float(), image.float())

            prediction = softmax(out)
            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                num[label[i]] += 1.0  # what is label[i]
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
    return sum(acc) / sum(num)


def main():
    args = get_arguments()
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    setup_seed(args.random_seed)

    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')

    model = AVClassifier(args)
    model.apply(weight_init)
    model.to(device)

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False)

    if args.train:
        trainloss_file = args.logs_path + '/test' + '/log.txt'
        if not os.path.exists(args.logs_path + '/test'):
            os.makedirs(args.logs_path + '/test')

        # save_path = args.ckpt_path + '/combine' + '/classifier-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
        #             str(args.batch_size) + '-lr' + str(args.learning_rate)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file)  # 删掉已有同名文件
        f_trainloss = open(trainloss_file, 'a')

        # load trained encoder

        load_path = args.load_path
        load_dict = torch.load(load_path)
        state_dict = load_dict['model']
        model.load_state_dict(state_dict)

        # load_path_v = args.load_path_v
        # load_dict_v = torch.load(load_path_v)
        # state_dict_v = load_dict_v['model']
        # visual_net.load_state_dict(state_dict_v)

        optimizer = optim.SGD(model.fusion_module.parameters(), lr=args.learning_rate, momentum=0.9,
                              weight_decay=1e-4)
        # optimizer_v = optim.SGD(visual_net.parameters(), lr=args.learning_rate, momentum=0.9,
        #                         weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

        best_acc = 0
        for epoch in range(args.epochs):
            print('Epoch: {}: '.format(epoch))

            batch_loss = train_uniclassifier_epoch(args, epoch, model, device,
                                                   train_dataloader, optimizer,
                                                   scheduler)
            acc = valid_uniclassifier(args, model, device, test_dataloader)
            print('epoch: ', epoch, 'loss: ', batch_loss, 'acc: ', acc)

            f_trainloss.write(str(epoch) +
                              "\t" + str(batch_loss) +
                              "\t" + str(acc) +
                              "\n")
            f_trainloss.flush()

            # if acc > best_acc or (epoch + 1) % args.epochs == 0:
            #     if acc > best_acc:
            #         best_acc = float(acc)
            #     print('Saving model....')
            #     # torch.save(
            #     #     {
            #     #         'classifier': classifier.state_dict(),
            #     #         'optimizer': optimizer.state_dict(),
            #     #         'scheduler': scheduler.state_dict()
            #     #     },
            #     #     os.path.join(save_path, 'epoch-{}.pt'.format(epoch))
            #     # )
            #     print('Saved model!!!')

        f_trainloss.close()


if __name__ == '__main__':
    main()
