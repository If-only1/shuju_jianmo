# -*- coding: utf-8 -*-
"""
@author: Li Xianyang
model define
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def act_function(x):
    return torch.sigmoid(x) * x


class Mlp(nn.Module):
    def __init__(self, hide_chanel=100):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(8, hide_chanel)
        self.fc2 = nn.Linear(hide_chanel, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = act_function(x)
        x = self.fc2(x)
        return x


class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.reset_parameters()

    def forward(self, X):
        Y = self.fc1(act_function(self.bn1(X)))
        Y = self.fc2(act_function(self.bn2(Y)))
        return Y + X

    def reset_parameters(self):
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', \
                                nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', \
                                nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)


class DenseBlock(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.reset_parameters()

    def forward(self, X):
        Y = self.fc1(act_function(self.bn1(X)))
        return torch.cat((Y, X), dim=1)

    def reset_parameters(self):
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', \
                                nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)


class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Head, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)

    def forward(self, X):
        Y = self.fc1(act_function(self.bn1(X)))
        return Y


def resfc(in_dim=3, out_dim=1, channels=32, hide_nums=3):
    net = nn.Sequential()
    net.add_module('first', nn.Linear(in_dim, channels))
    for i in range(hide_nums):
        net.add_module('res_block{}'.format(i), module=Residual(channels, channels))
    net.add_module('head', Head(channels, out_dim))
    return net


def densefc(in_dim=3, out_dim=1, channels=32, hide_nums=3):
    net = nn.Sequential()
    net.add_module('first', nn.Linear(in_dim, channels))
    in_channels = channels
    for i in range(hide_nums):
        net.add_module('block{}'.format(i), module=DenseBlock(in_channels, channels))
        in_channels += channels
    net.add_module('head', Head(in_channels, out_dim))
    return net


if __name__ == '__main__':
    res = densefc(3, 3)
    res.eval()
    input = torch.randn(1, 3)
    # torch.save(res,'./model.pth')
    torch.onnx.export(res, input, './model.onnx')
    # print(res)

    test = torch.randn((3, 3))
    print(res(test))
