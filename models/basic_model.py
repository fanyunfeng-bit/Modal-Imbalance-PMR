import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18, resnet34, resnet101
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion


class AClassifier(nn.Module):
    def __init__(self, args):
        super(AClassifier, self).__init__()
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

        self.net = resnet18(modality='audio')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, audio):
        a = self.net(audio)
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1)
        out = self.classifier(a)
        return out


class VClassifier(nn.Module):
    def __init__(self, args):
        super(VClassifier, self).__init__()
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

        self.net = resnet18(modality='visual')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, visual, B):
        v = self.net(visual)
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        out = self.classifier(v)
        return out


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
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

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out


class AVClassifier_34(nn.Module):
    def __init__(self, args):
        super(AVClassifier_34, self).__init__()

        fusion = args.fusion_method
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

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet34(modality='audio')
        self.visual_net = resnet34(modality='visual')

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        # print('concat: ', v.shape)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        # print('dis: ', v.shape)
        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out


class AVClassifier_101(nn.Module):
    def __init__(self, args):
        super(AVClassifier_101, self).__init__()

        fusion = args.fusion_method
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

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet101(modality='audio')
        self.visual_net = resnet101(modality='visual')

    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        # print('concat: ', v.shape)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        # print('dis: ', v.shape)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)
        # print('avg: ', v.shape)
        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out


class CLClassifier(nn.Module):
    def __init__(self, args):
        super(CLClassifier, self).__init__()

        self.fusion = args.fusion_method
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

        if self.fusion == 'concat':
            self.fc_out = nn.Linear(args.embed_dim * 2, n_classes)
        elif self.fusion == 'sum':
            self.fc_x = nn.Linear(args.embed_dim, n_classes)
            self.fc_y = nn.Linear(args.embed_dim, n_classes)

    def forward(self, x, y):
        if self.fusion == 'concat':
            output = torch.cat((x, y), dim=1)
            output = self.fc_out(output)
        return output


# Colored-and-gray-MNIST
class convnet(nn.Module):
    def __init__(self, num_classes=10, modal='gray'):
        super(convnet, self).__init__()

        self.modal = modal

        if modal == 'gray':
            in_channel = 1
        elif modal == 'colored':
            in_channel = 3
        else:
            raise ValueError('non exist modal')
        self.bn0 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64, 512)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x)  # 14x14

        x = self.conv2(x)
        x = self.relu(x)  # 14x14
        x = self.conv3(x)
        x = self.relu(x)  # 7x7
        x = self.conv4(x)
        x = self.relu(x)  # 7x7

        feat = x
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)

        return feat


class CGClassifier(nn.Module):
    def __init__(self, args):
        super(CGClassifier, self).__init__()

        fusion = args.fusion_method

        n_classes = 10

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.gray_net = convnet(modal='gray')
        self.colored_net = convnet(modal='colored')

    def forward(self, gray, colored):
        g = self.gray_net(gray)
        c = self.colored_net(colored)

        g = torch.flatten(g, 1)
        c = torch.flatten(c, 1)

        g, c, out = self.fusion_module(g, c)
        return g, c, out


class GrayClassifier(nn.Module):
    def __init__(self, args):
        super(GrayClassifier, self).__init__()
        if args.dataset == 'CGMNIST':
            n_classes = 10

        self.net = convnet(modal='gray')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, gray):
        g = self.net(gray)
        g = torch.flatten(g, 1)
        g_out = self.classifier(g)
        return g_out


class ColoredClassifier(nn.Module):
    def __init__(self, args):
        super(ColoredClassifier, self).__init__()
        if args.dataset == 'CGMNIST':
            n_classes = 10

        self.net = convnet(modal='colored')
        self.classifier = nn.Linear(args.embed_dim, n_classes)

    def forward(self, color):
        c = self.net(color)
        c = torch.flatten(c, 1)
        c_out = self.classifier(c)
        return c_out


