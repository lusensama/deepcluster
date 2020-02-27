# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from random import random as rd
__all__ = [ 'VGG', 'vgg16', 'vgg15ab', 'lenet']

class LeNet(nn.Module):
    def __init__(self,  num_classes=10, sobel=False):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            # input channel = 1, output channel = 6, kernel_size = 5
            # input size = (32, 32), output size = (28, 28)
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # input channel = 6, output channel = 16, kernel_size = 5
            # input size = (14, 14), output size = (10, 10)
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.classifier = nn.Sequential(
            # input dim = 16*5*5, output dim = 120
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            # input dim = 120, output dim = 84
            nn.Linear(120, 84),
            nn.ReLU()
        )
        # input dim = 84, output dim = 10
        self.top_layer = nn.Linear(84, num_classes)
        self._initialize_weights()
        if sobel:
            grayscale = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0,0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1,0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        # flatten as one dimension
        x = x.view(x.size(0), -1)
        # input dim = 16*5*5, output dim = 120
        x = self.classifier(x)
        # input dim = 84, output dim = 10
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                #print(y)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def lenet(dataset='mnist', **kwargs):
    if dataset == 'mnist':
        model = LeNet()
    else:
        raise ValueError('Unsupported Dataset!')
    return model

class VGG(nn.Module):

    def __init__(self, features, num_classes, sobel):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )
        self.top_layer = nn.Linear(4096, num_classes)
        self._initialize_weights()
        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0,0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1,0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(input_dim, batch_norm):
    layers = []
    in_channels = input_dim
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(sobel=False, bn=True, out=1000):
    dim = 2 + int(not sobel)
    model = VGG(make_layers(dim, bn), out, sobel)
    return model




class VGG_15_avg_before_relu(nn.Module):
    def __init__(self,  dr=0.1, num_classes=1000, units=512*7*7, sobel=False):
        super(VGG_15_avg_before_relu, self).__init__()
        self.features = nn.Sequential(
            # in_channels 1, out_channels 64, kernel_size 3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dr),
            nn.Linear(units, 4096, bias=False),  # Linear,
            nn.ReLU(),
            nn.Dropout(dr),

        )
        self.top_layer = nn.Linear(4096, num_classes, bias=False)  # Linear,
        self._initialize_weights()
        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

def vgg15ab(dataset='imagenet', **kwargs):
    if dataset == 'imagenet':
        model = VGG_15_avg_before_relu(num_classes=1000, **kwargs)
    elif dataset == 'cifar100':
        model = VGG_15_avg_before_relu(num_classes=100, units=512,**kwargs)
    elif dataset == 'mnist':
        model = VGG_15_avg_before_relu(num_classes=10, units=512, **kwargs)
    else:
        raise ValueError('Unsupported Dataset!')
    return model
