import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.CGMNIST import CGMNISTDataset
from dataset.CramedDataset import CramedDataset
from dataset.AVEDataset import AVEDataset
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier, CGClassifier
from utils.utils import setup_seed, weight_init

from dataset.VGGSoundDataset import VGGSound


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE', 'Acc'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int, help='Extract how many frames in a second')
    parser.add_argument('--num_frame', default=3, type=int, help='use how many frames for train')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)

    parser.add_argument('--learning_rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', required=True, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', default='ckpt', type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', action='store_true', help='whether to visualize')
    parser.add_argument('--logs_path', default='logs', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--gpu', type=int, default=0)  # gpu
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    # args = parser.parse_args()
    #
    # args.use_cuda = torch.cuda.is_available() and not args.no_cuda

    return parser.parse_args()


def dot_product_angle_tensor(v1, v2):
    vector_dot_product = torch.dot(v1, v2)
    arccos = torch.acos(vector_dot_product / (torch.norm(v1, p=2) * torch.norm(v2, p=2)))
    angle = np.degrees(arccos.data.cpu().numpy())
    return arccos, angle


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, ratio_a, writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    _a_angle = 0
    _v_angle = 0
    _ratio_a = 0

    # angle_file = args.logs_path + '/Method-CE-grad' + '/angle-' + args.dataset + '-' + args.fusion_method + '-bsz' + \
    #              str(args.batch_size) + '-lr' + str(args.learning_rate) \
    #              + '-epoch' + str(args.epochs) + '-' + args.modulation + str(args.alpha) + \
    #                      '-ends' + str(args.modulation_ends) + '.txt'
    # f_angle = open(angle_file, 'a')

    for step, (spec, image, label) in enumerate(dataloader):

        spec = spec.to(device)  # B x 257 x 1004(CREMAD 299)
        image = image.to(device)  # B x 1(image count) x 3 x 224 x 224
        label = label.to(device)  # B

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        if args.dataset != 'CGMNIST':
            a, v, out = model(spec.unsqueeze(1).float(), image.float())
        else:
            a, v, out = model(spec, image)  # gray colored

        # out = softmax(out)

        if args.fusion_method == 'sum':
            out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                     model.fusion_module.fc_y.bias)
            out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                     model.fusion_module.fc_x.bias)
        elif args.fusion_method == 'concat':
            weight_size = model.fusion_module.fc_out.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                     + model.fusion_module.fc_out.bias / 2)
            out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                     + model.fusion_module.fc_out.bias / 2)
        elif args.fusion_method == 'film' or args.fusion_method == 'gated':
            out_v = out
            out_a = out

        loss = criterion(out, label)
        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)

        # grad_a = torch.Tensor([]).to(device)
        # grad_v = torch.Tensor([]).to(device)
        # grad_a_fusion = torch.Tensor([]).to(device)
        # grad_v_fusion = torch.Tensor([]).to(device)
        #
        # loss_v.backward(retain_graph=True)
        # for parms in model.visual_net.parameters():
        #     grad_v = torch.cat((grad_v, parms.grad.flatten()), 0)
        # optimizer.zero_grad()
        #
        # loss_a.backward(retain_graph=True)
        # for parms in model.audio_net.parameters():
        #     grad_a = torch.cat((grad_a, parms.grad.flatten()), 0)
        # optimizer.zero_grad()

        loss.backward()
        # for parms in model.audio_net.parameters():
        #     grad_a_fusion = torch.cat((grad_a_fusion, parms.grad.flatten()), 0)
        # for parms in model.visual_net.parameters():
        #     grad_v_fusion = torch.cat((grad_v_fusion, parms.grad.flatten()), 0)
        #
        # print('loss: ', loss.data, 'loss_v: ', loss_v.data, 'loss_a: ', loss_a.data)
        # #
        # # calculate the angle
        # _, a_angle = dot_product_angle_tensor(grad_a, grad_a_fusion)
        # _, v_angle = dot_product_angle_tensor(grad_v, grad_v_fusion)
        # _a_angle += a_angle
        # _v_angle += v_angle

        if args.modulation == 'Normal':
            pass
        else:
            ratio_v = 1 / ratio_a

            """
            Below is the Eq.(10) in our CVPR paper:
                    1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            k_t_u =
                    1,                         else
            coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """
            if ratio_v > 1:
                coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
                coeff_a = 1
                acc_v = 1
                acc_a = 1 + tanh(args.alpha * relu(ratio_v))
            else:
                coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
                coeff_v = 1
                acc_a = 1
                acc_v = 1 + tanh(args.alpha * relu(ratio_a))

            if args.modulation_starts <= epoch <= args.modulation_ends:  # bug fixed
                if args.dataset != 'CGMNIST':
                    for name, parms in model.named_parameters():
                        layer = str(name).split('.')[0]

                        if 'audio' in layer and len(parms.grad.size()) == 4:
                            if args.modulation == 'OGM_GE':  # bug fixed
                                parms.grad = parms.grad * coeff_a + \
                                             torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                            elif args.modulation == 'OGM':
                                parms.grad *= coeff_a
                            elif args.modulation == 'Acc':
                                parms.grad *= acc_a

                        if 'visual' in layer and len(parms.grad.size()) == 4:
                            if args.modulation == 'OGM_GE':  # bug fixed
                                parms.grad = parms.grad * coeff_v + \
                                             torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                            elif args.modulation == 'OGM':
                                parms.grad *= coeff_v
                            elif args.modulation == 'Acc':
                                parms.grad *= acc_v
                else:
                    for name, parms in model.named_parameters():
                        layer = str(name).split('.')[0]
                        if 'gray' in layer and len(parms.grad.size()) == 4:
                            if args.modulation == 'OGM_GE':  # bug fixed
                                parms.grad = parms.grad * coeff_a + \
                                             torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                            elif args.modulation == 'OGM':
                                parms.grad *= coeff_a
                            elif args.modulation == 'Acc':
                                parms.grad *= acc_a

                        if 'colored' in layer and len(parms.grad.size()) == 4:
                            if args.modulation == 'OGM_GE':  # bug fixed
                                parms.grad = parms.grad * coeff_v + \
                                             torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                            elif args.modulation == 'OGM':
                                parms.grad *= coeff_v
                            elif args.modulation == 'Acc':
                                parms.grad *= acc_v
            else:
                pass

        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()
        _ratio_a += ratio_a

        print('loss: ', loss.item(), loss_a.item(), loss_v.item())

        # f_angle.write(str(ratio_a) +
        #               "\t" + str(a_angle) +
        #               "\t" + str(v_angle) +
        #               "\n")
        # f_angle.flush()

    scheduler.step()
    # f_angle.close()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _a_angle / len(dataloader), \
           _v_angle / len(dataloader), _ratio_a / len(dataloader)


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'CGMNIST':
        n_classes = 10
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if args.dataset != 'CGMNIST':
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec, image)  # gray colored

            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                         model.fusion_module.fc_y.bias)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                         model.fusion_module.fc_x.bias)
            elif args.fusion_method == 'concat':
                weight_size = model.fusion_module.fc_out.weight.size(1)
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                         + model.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                         + model.fusion_module.fc_out.bias / 2)
            elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                out_v = out
                out_a = out

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def fine_tuning_classifier(args, model, device, train_dataloader, test_dataloader, a_classifier, v_classifier,
                           ac_optimizer, vc_optimizer):
    model.eval()
    a_classifier.train()
    v_classifier.train()

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'CGMNIST':
        n_classes = 10
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    for e in range(1):
        for step, (spec, image, label) in enumerate(train_dataloader):

            spec = spec.to(device)  # B x 257 x 1004(CREMAD 299)
            image = image.to(device)  # B x 1(image count) x 3 x 224 x 224
            label = label.to(device)  # B

            ac_optimizer.zero_grad()
            vc_optimizer.zero_grad()
            with torch.no_grad():
                # TODO: make it simpler and easier to extend
                if args.dataset != 'CGMNIST':
                    a, v, out = model(spec.unsqueeze(1).float(), image.float())
                else:
                    a, v, out = model(spec, image)  # gray colored

            a_out = a_classifier(a)
            v_out = v_classifier(v)

            loss_a = criterion(a_out, label)
            loss_v = criterion(v_out, label)

            loss_a.backward()
            loss_v.backward()

            ac_optimizer.step()
            vc_optimizer.step()

    a_classifier.eval()
    v_classifier.eval()

    _score_v = 0
    _score_a = 0
    num = [0.0 for _ in range(n_classes)]
    acc_a = [0.0 for _ in range(n_classes)]
    acc_v = [0.0 for _ in range(n_classes)]
    for step, (spec, image, label) in enumerate(test_dataloader):
        spec = spec.to(device)  # B x 257 x 1004(CREMAD 299)
        image = image.to(device)  # B x 1(image count) x 3 x 224 x 224
        label = label.to(device)  # B

        with torch.no_grad():
            # TODO: make it simpler and easier to extend
            if args.dataset != 'CGMNIST':
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec, image)  # gray colored

            a_out = a_classifier(a)
            v_out = v_classifier(v)

            score_v = sum([softmax(v_out)[i][label[i]] for i in range(v_out.size(0))])
            score_a = sum([softmax(a_out)[i][label[i]] for i in range(a_out.size(0))])

            _score_a += score_a
            _score_v += score_v

            pred_v = softmax(v_out)
            pred_a = softmax(a_out)

            for i in range(image.shape[0]):
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0
    _score_v /= len(test_dataloader)
    _score_a /= len(test_dataloader)
    ratio_a = _score_a / _score_v

    return ratio_a, sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():
    args = get_arguments()
    args.use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    setup_seed(args.random_seed)

    device = torch.device('cuda:' + str(args.gpu) if args.use_cuda else 'cpu')

    if args.dataset == 'CGMNIST':
        model = CGClassifier(args)
    else:
        model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    # model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

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
        train_dataset = AVEDataset(args, mode='train')
        test_dataset = AVEDataset(args, mode='test')
    elif args.dataset == 'CGMNIST':
        train_dataset = CGMNISTDataset(args, mode='train')
        test_dataset = CGMNISTDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=False)  # 计算机的内存充足的时候，可以设置pin_memory=True

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False)

    if args.train:

        trainloss_file = args.logs_path + '/Method-CE-grad' + '/train_loss-' + args.dataset + '-' + args.fusion_method + '-False-bsz' + \
                         str(args.batch_size) + '-lr' + str(args.learning_rate) \
                         + '-epoch' + str(args.epochs) + '-' + args.modulation + str(args.alpha) + \
                         '-ends' + str(args.modulation_ends) + '.txt'
        if not os.path.exists(args.logs_path + '/Method-CE-grad'):
            os.makedirs(args.logs_path + '/Method-CE-grad')

        save_path = args.ckpt_path + '/Method-CE-grad' + '/model-' + args.dataset + '-' + args.fusion_method + '-False-bsz' + \
                    str(args.batch_size) + '-lr' + str(args.learning_rate) \
                    + '-epoch' + str(args.epochs) + '-' + args.modulation + str(args.alpha) + \
                         '-ends' + str(args.modulation_ends)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if (os.path.isfile(trainloss_file)):
            os.remove(trainloss_file)  # 删掉已有同名文件
        f_trainloss = open(trainloss_file, 'a')

        best_acc = 0.0

        ratio_a = torch.Tensor([1]).to(device)

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            batch_loss, batch_loss_a, batch_loss_v, a_angle, v_angle, ratio_a = train_epoch(args, epoch, model, device,
                                                                                   train_dataloader, optimizer,
                                                                                   scheduler, ratio_a)

            acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

            if epoch % 3 == 0 and args.modulation != 'Normal':
                a_classifier = copy.deepcopy(model.fusion_module.fc_out)
                v_classifier = copy.deepcopy(model.fusion_module.fc_out)

                ac_optimizer = optim.SGD(a_classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
                vc_optimizer = optim.SGD(v_classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
                ratio_a, acc_a, acc_v = fine_tuning_classifier(args, model, device, train_dataloader, test_dataloader, a_classifier, v_classifier,
                                                               ac_optimizer, vc_optimizer)

            print('epoch: ', epoch, 'loss: ', batch_loss, batch_loss_a, batch_loss_v)
            print('epoch: ', epoch, 'acc: ', acc, 'acc_a: ', acc_a, 'acc_v: ', acc_v)
            print('epoch: ', epoch, 'a_angle: ', a_angle, 'v_angle: ', v_angle)
            f_trainloss.write(str(epoch) +
                              "\t" + str(batch_loss) +
                              "\t" + str(batch_loss_a) +
                              "\t" + str(batch_loss_v) +
                              "\t" + str(acc) +
                              "\t" + str(acc_a) +
                              "\t" + str(acc_v) +
                              "\t" + str(a_angle) +
                              "\t" + str(v_angle) +
                              "\t" + str(ratio_a) +
                              "\n")
            f_trainloss.flush()

            if acc > best_acc or (epoch + 1) % 10 == 0:
                if acc > best_acc:
                    best_acc = float(acc)

                print('Saving model....')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    },
                    os.path.join(save_path, 'epoch-{}.pt'.format(epoch))
                )
                print('Saved model!!!')
        f_trainloss.close()

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
