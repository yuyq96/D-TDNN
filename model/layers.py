import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _single, _pair


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for i, name in enumerate(config_str.split('-')):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


def statistics_pooling(x, order=2, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    stats = []
    mean = x.mean(dim=dim)
    stats.append(mean)
    if order >= 2:
        std = x.std(dim=dim, unbiased=unbiased)
        stats.append(std)
    if order >= 3:
        x = (x - mean.unsqueeze(-1)) / std.clamp(min=eps).unsqueeze(-1)
        skewness = x.pow(3).mean(-1)
        stats.append(skewness)
        if order >= 4:
            kurtosis = x.pow(4).mean(-1)
            stats.append(kurtosis)
    stats = torch.cat(stats, dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):

    def __init__(self, order=2):
        super(StatsPool, self).__init__()
        self.order = order

    def forward(self, x):
        return statistics_pooling(x, order=self.order)


class TimeDelay(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(TimeDelay, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _pair(padding)
        self.dilation = _single(dilation)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels * kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            std = 1 / math.sqrt(self.out_channels)
            self.weight.normal_(0, std)
            if self.bias is not None:
                self.bias.normal_(0, std)

    def forward(self, x):
        x = F.pad(x, self.padding).unsqueeze(1)
        x = F.unfold(x, (self.in_channels,)+self.kernel_size, dilation=(1,)+self.dilation, stride=(1,)+self.stride)
        return F.linear(x.transpose(1, 2), self.weight, self.bias).transpose(1, 2)


class TDNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True, config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = TimeDelay(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class DenseTDNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bn_channels, kernel_size, stride=1,
                 dilation=1, bias=False, config_str='batchnorm-relu'):
        super(DenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Linear(in_channels, bn_channels, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.linear2 = TimeDelay(bn_channels, out_channels, kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.linear1(self.nonlinear1(x).transpose(1, 2)).transpose(1, 2)
        x = self.linear2(self.nonlinear2(x))
        return x


class DenseTDNNBlock(nn.ModuleList):

    def __init__(self, num_layers, in_channels, out_channels, bn_channels, kernel_size,
                 stride=1, dilation=1, bias=False, config_str='batchnorm-relu'):
        super(DenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str
            )
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], 1)
        return x


class TransitLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x.transpose(1, 2)).transpose(1, 2)
        return x


class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x)
        else:
            x = self.linear(x.transpose(1, 2)).transpose(1, 2)
        x = self.nonlinear(x)
        return x
