#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
import collections
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import *
from torch.optim import SGD


# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
# from ranger import RangerQH
# from ranger import RangerVA
# from ranger import Ranger
# from ranger21 import Ranger21


def get_optimizer(model, lr, weight_decay=0, nesterov=True):
    if weight_decay != 0:
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, (nn.BatchNorm2d, nn.LayerNorm)):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)

        opt = SGD(g0, lr, 0.9, nesterov=nesterov)
        """
        opt = RangerVA(g0, lr)
        opt = RangerQH(g0, lr)
        opt = Ranger(g0, lr)
        """
        opt.add_param_group({'params': g1, 'weight_decay': weight_decay})  # add g1 with weight_decay
        opt.add_param_group({'params': g2})  # add g2 (biases)
    else:
        opt = SGD(model.parameters(), lr, 0.9, nesterov=nesterov)
        """
        opt = RangerVA(model.parameters(), lr)
        opt = RangerQH(model.parameters(), lr)
        opt = Ranger(model.parameters(), lr)
        """
    return opt


def pad_num(k_s):
    pad_per_side = int((k_s - 1) * 0.5)
    return pad_per_side


class SE(nn.Module):
    def __init__(self, cin, ratio):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(cin, int(cin / ratio), bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1.out_features, cin, bias=False)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1, x.size()[1])
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = x.view(-1, x.size()[1], 1, 1)
        return x * y


class SE_LN(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1, 1))
        self.ln = nn.LayerNorm(cin)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1, x.size(1))
        x = self.ln(x)
        x = self.act(x)
        x = x.view(-1, x.size(1), 1, 1)
        return x * y


class DFSEBV1(nn.Module):
    def __init__(self, cin, dw_s, ratio, is_LN):
        super().__init__()
        self.pw1 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.ReLU()
        self.dw1 = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)
        self.act2 = nn.Hardswish()
        if is_LN:
            self.se1 = SE_LN(cin)
        else:
            self.se1 = SE(cin, 3)

        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act3 = nn.ReLU()
        self.dw2 = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)
        self.act4 = nn.Hardswish()
        if is_LN:
            self.se2 = SE_LN(cin)
        else:
            self.se2 = SE(cin, 3)

    def forward(self, x):
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.act2(x)
        x = self.se1(x)
        x = x + y

        x = self.pw2(x)
        x = self.bn2(x)
        x = self.act3(x)
        x = self.dw2(x)
        x = self.act4(x)
        x = self.se2(x)
        x = x + y
        # del y
        return x


class DFSEBV2(nn.Module):
    def __init__(self, cin, dw_s, is_LN):
        super().__init__()
        self.pw1 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.SiLU()
        self.dw1 = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)
        if is_LN:
            self.seln = SE_LN(cin)
        else:
            self.seln = SE(cin, 3)

        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act2 = nn.Hardswish()
        self.dw2 = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)

    def forward(self, x):
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.seln(x)
        x += y

        x = self.pw2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dw2(x)
        x += y
        return x


# Feature concentrator
class FCT(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 4, 2, 1, groups=cin, bias=False)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.minpool = MinPool2d(2, ceil_mode=True)
        self.pw = nn.Conv2d(3 * cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x),
            self.dw(x),
        ), 1)
        x = self.pw(x)
        x = self.bn(x)
        return x


class EVE(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.minpool = MinPool2d(2, ceil_mode=True)
        self.pw = nn.Conv2d(2 * cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x)
        ), 1)
        x = self.pw(x)
        x = self.bn(x)
        return x


class ME(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.pw = nn.Conv2d(cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.pw(x)
        x = self.bn(x)
        return x


class MinPool2d(nn.Module):
    def __init__(self, ks, ceil_mode):
        super().__init__()
        self.ks = ks
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return -F.max_pool2d(-x, self.ks, ceil_mode=self.ceil_mode)


class DW(nn.Module):
    def __init__(self, cin, dw_s):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.dw(x)
        x = self.act(x)
        return x


class ExquisiteNetV1(nn.Module):
    def __init__(self, class_num, img_channels):
        super().__init__()
        self.features = nn.Sequential(
            collections.OrderedDict([
                ('ME1', ME(img_channels, 12)),
                ('DFSEB1', DFSEBV1(12, 3, 3, False)),

                ('ME2', ME(12, 50)),
                ('DFSEB2', DFSEBV1(50, 3, 3, False)),

                ('ME3', ME(50, 100)),
                ('DFSEB3', DFSEBV1(100, 3, 3, False)),

                ('ME4', ME(100, 200)),
                ('DFSEB4', DFSEBV1(200, 3, 3, False)),

                ('ME5', ME(200, 350)),
                ('DFSEB5', DFSEBV1(350, 3, 3, False)),

                ('conv', nn.Conv2d(350, 640, 1, 1)),
                ('act', nn.Hardswish())
            ])
        )
        self.gavg = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(640, class_num)

    def forward(self, x):
        x = self.features(x)
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1, x.size()[1])
        x = self.fc(x)
        return x


class ExquisiteNetV2(nn.Module):
    def __init__(self, class_num, img_channels):
        super().__init__()
        self.FCT = FCT(img_channels, 12)
        self.DFSEB1 = DFSEBV2(12, 3, True)  #
        self.EVE = EVE(12, 48)
        self.DFSEB2 = DFSEBV2(48, 3, True)  #
        self.ME3 = ME(48, 96)
        self.DFSEB3 = DFSEBV2(96, 3, True)  #
        self.ME4 = ME(96, 192)
        self.DFSEB4 = DFSEBV2(192, 3, True)  #
        self.ME5 = ME(192, 384)
        self.DFSEB5 = DFSEBV2(384, 3, True)  #
        self.DW = DW(384, 3)  #
        self.gavg = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(384, class_num)

    def forward(self, x):
        x = self.FCT(x)
        x = self.DFSEB1(x)  #
        x = self.EVE(x)
        x = self.DFSEB2(x)  #
        x = self.ME3(x)
        x = self.DFSEB3(x)  #
        x = self.ME4(x)
        x = self.DFSEB4(x)  #
        x = self.ME5(x)
        x = self.DFSEB5(x)  #
        x = self.DW(x)  #
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x


import collections
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import *
from torch.optim import SGD


# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
# from ranger import RangerQH
# from ranger import RangerVA
# from ranger import Ranger
# from ranger21 import Ranger21


def get_optimizer(model, lr, weight_decay=0, nesterov=True):
    if weight_decay != 0:
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, (nn.BatchNorm2d, nn.LayerNorm)):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)

        opt = SGD(g0, lr, 0.9, nesterov=nesterov)
        """
        opt = RangerVA(g0, lr)
        opt = RangerQH(g0, lr)
        opt = Ranger(g0, lr)
        """
        opt.add_param_group({'params': g1, 'weight_decay': weight_decay})  # add g1 with weight_decay
        opt.add_param_group({'params': g2})  # add g2 (biases)
    else:
        opt = SGD(model.parameters(), lr, 0.9, nesterov=nesterov)
        """
        opt = RangerVA(model.parameters(), lr)
        opt = RangerQH(model.parameters(), lr)
        opt = Ranger(model.parameters(), lr)
        """
    return opt


def pad_num(k_s):
    pad_per_side = int((k_s - 1) * 0.5)
    return pad_per_side


class SE(nn.Module):
    def __init__(self, cin, ratio):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(cin, int(cin / ratio), bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1.out_features, cin, bias=False)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1, x.size()[1])
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = x.view(-1, x.size()[1], 1, 1)
        return x * y


class SE_LN(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.gavg = nn.AdaptiveAvgPool2d((1, 1))
        self.ln = nn.LayerNorm(cin)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = x
        x = self.gavg(x)
        x = x.view(-1, x.size(1))
        x = self.ln(x)
        x = self.act(x)
        x = x.view(-1, x.size(1), 1, 1)
        return x * y


class DFSEBV1(nn.Module):
    def __init__(self, cin, dw_s, ratio, is_LN):
        super().__init__()
        self.pw1 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.ReLU()
        self.dw1 = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)
        self.act2 = nn.Hardswish()
        if is_LN:
            self.se1 = SE_LN(cin)
        else:
            self.se1 = SE(cin, 3)

        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act3 = nn.ReLU()
        self.dw2 = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)
        self.act4 = nn.Hardswish()
        if is_LN:
            self.se2 = SE_LN(cin)
        else:
            self.se2 = SE(cin, 3)

    def forward(self, x):
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.act2(x)
        x = self.se1(x)
        x = x + y

        x = self.pw2(x)
        x = self.bn2(x)
        x = self.act3(x)
        x = self.dw2(x)
        x = self.act4(x)
        x = self.se2(x)
        x = x + y
        # del y
        return x


class DFSEBV2(nn.Module):
    def __init__(self, cin, dw_s, is_LN):
        super().__init__()
        self.pw1 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.act1 = nn.SiLU()
        self.dw1 = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)
        if is_LN:
            self.seln = SE_LN(cin)
        else:
            self.seln = SE(cin, 3)

        self.pw2 = nn.Conv2d(cin, cin, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cin)
        self.act2 = nn.Hardswish()
        self.dw2 = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)

    def forward(self, x):
        y = x
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.seln(x)
        x += y

        x = self.pw2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dw2(x)
        x += y
        return x


# Feature concentrator
class FCT(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 4, 2, 1, groups=cin, bias=False)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.minpool = MinPool2d(2, ceil_mode=True)
        self.pw = nn.Conv2d(3 * cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x),
            self.dw(x),
        ), 1)
        x = self.pw(x)
        x = self.bn(x)
        return x


class EVE(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.minpool = MinPool2d(2, ceil_mode=True)
        self.pw = nn.Conv2d(2 * cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = torch.cat((
            self.maxpool(x),
            self.minpool(x)
        ), 1)
        x = self.pw(x)
        x = self.bn(x)
        return x


class ME(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.pw = nn.Conv2d(cin, cout, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.pw(x)
        x = self.bn(x)
        return x


class MinPool2d(nn.Module):
    def __init__(self, ks, ceil_mode):
        super().__init__()
        self.ks = ks
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return -F.max_pool2d(-x, self.ks, ceil_mode=self.ceil_mode)


class DW(nn.Module):
    def __init__(self, cin, dw_s):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, dw_s, 1, pad_num(dw_s), groups=cin)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.dw(x)
        x = self.act(x)
        return x


class ExquisiteNetV1(nn.Module):
    def __init__(self, class_num, img_channels):
        super().__init__()
        self.features = nn.Sequential(
            collections.OrderedDict([
                ('ME1', ME(img_channels, 12)),
                ('DFSEB1', DFSEBV1(12, 3, 3, False)),

                ('ME2', ME(12, 50)),
                ('DFSEB2', DFSEBV1(50, 3, 3, False)),

                ('ME3', ME(50, 100)),
                ('DFSEB3', DFSEBV1(100, 3, 3, False)),

                ('ME4', ME(100, 200)),
                ('DFSEB4', DFSEBV1(200, 3, 3, False)),

                ('ME5', ME(200, 350)),
                ('DFSEB5', DFSEBV1(350, 3, 3, False)),

                ('conv', nn.Conv2d(350, 640, 1, 1)),
                ('act', nn.Hardswish())
            ])
        )
        self.gavg = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(640, class_num)

    def forward(self, x):
        x = self.features(x)
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1, x.size()[1])
        x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
        return x


class ExquisiteNetV2(nn.Module):
    def __init__(self, class_num, img_channels):
        super().__init__()
        self.FCT = FCT(img_channels, 12)
        self.DFSEB1 = DFSEBV2(12, 3, True)  #
        self.EVE = EVE(12, 48)
        self.DFSEB2 = DFSEBV2(48, 3, True)  #
        self.ME3 = ME(48, 96)
        self.DFSEB3 = DFSEBV2(96, 3, True)  #
        self.ME4 = ME(96, 192)
        self.DFSEB4 = DFSEBV2(192, 3, True)  #
        self.ME5 = ME(192, 384)
        self.DFSEB5 = DFSEBV2(384, 3, True)  #
        self.DW = DW(384, 3)  #
        self.gavg = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(384, class_num)

    def forward(self, x):
        x = self.FCT(x)
        x = self.DFSEB1(x)  #
        x = self.EVE(x)
        x = self.DFSEB2(x)  #
        x = self.ME3(x)
        x = self.DFSEB3(x)  #
        x = self.ME4(x)
        x = self.DFSEB4(x)  #
        x = self.ME5(x)
        x = self.DFSEB5(x)  #
        x = self.DW(x)  #
        x = self.gavg(x)
        x = self.drop(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits
