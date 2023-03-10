import copy
import csv
import os
import pickle
# import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import pdb

class CGMNISTDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        # self.image_gray = []
        # self.image_color = []
        # self.label = []
        self.mode = mode
        # classes = []

        if self.mode == 'train':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
            self.gray_dataset = datasets.MNIST(root='data/Mnist/mnist/', train=True, download=True,
                                               transform=transform)

        else:
            transform = transforms.Compose([
                # transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])
            self.gray_dataset = datasets.MNIST(root='data/Mnist/mnist/', train=False, download=True,
                                               transform=transform)

        # colored MNIST
        data_dic = np.load('data/Mnist/colored_mnist/mnist_10color_jitter_var_%.03f.npy' % 0.030,
                           encoding='latin1', allow_pickle=True).item()
        if self.mode == 'train':
            self.colored_image = data_dic['train_image']
            self.colored_label = data_dic['train_label']
        elif self.mode == 'test':
            self.colored_image = data_dic['test_image']
            self.colored_label = data_dic['test_label']

        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.ToPIL = transforms.Compose([
            transforms.ToPILImage(),
        ])


    def __len__(self):
        return len(self.gray_dataset)

    def __getitem__(self, idx):
        gray_image, gray_label = self.gray_dataset[idx]

        colored_label = self.colored_label[idx]
        colored_image = self.colored_image[idx]

        colored_image = self.ToPIL(colored_image)

        return gray_image, self.T(colored_image), gray_label

# args = 0
# data = CGMNISTDataset(args)
# count = 0
# for gray_i, colored_i, colored_l in data:
#     if gray_i.shape[1] != colored_i.shape[1]:
#         print('labels are different')
#     count += 1
# print('finished', count)



