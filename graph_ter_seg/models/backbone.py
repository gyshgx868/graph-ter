import torch

import torch.nn as nn

from collections import OrderedDict

from graph_ter_seg.models.layers import EdgeConvolution
from graph_ter_seg.models.layers import MultiEdgeConvolution
from graph_ter_seg.tools import utils


class Encoder(nn.Module):
    def __init__(self, k=20):
        super(Encoder, self).__init__()
        self.conv0 = MultiEdgeConvolution(k, in_features=3, mlp=(64, 64))
        self.conv1 = MultiEdgeConvolution(k, in_features=64, mlp=(64, 64))
        self.conv2 = EdgeConvolution(k, in_features=64, out_features=64)
        self.conv3 = EdgeConvolution(k, in_features=64, out_features=64)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        features = torch.cat((x1, x2, x3, x4), dim=1)
        return features


class Tail(nn.Module):
    def __init__(self, in_features=256):
        super(Tail, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(in_features, 128, kernel_size=1, bias=False)),
            ('bn0', nn.BatchNorm1d(128)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2))
        ]))

    def forward(self, x):
        x = self.conv(x)
        return x


class Backbone(nn.Module):
    def __init__(self, k=20, out_features=3):
        super(Backbone, self).__init__()
        self.encoder = Encoder(k=k)
        self.tail = Tail(in_features=256)
        self.decoder = nn.Sequential(OrderedDict([
            ('conv0', EdgeConvolution(k, in_features=256, out_features=128)),
            ('conv1', EdgeConvolution(k, in_features=128, out_features=64)),
            ('conv2', nn.Conv1d(64, out_features, kernel_size=1))
        ]))

    def forward(self, *args):
        if len(args) == 2:
            x, y = args[0], args[1]
            x1 = self.tail(self.encoder(x))
            x2 = self.tail(self.encoder(y))
            x = torch.cat((x1, x2), dim=1)  # B * 2F * N
            matrix = self.decoder(x)
            return matrix
        elif len(args) == 1:
            x = args[0]
            features = self.encoder(x)
            return features
        else:
            raise ValueError('Invalid number of arguments.')


def main():
    encoder = Encoder()
    encoder_para = utils.get_total_parameters(encoder)
    print('Encoder:', encoder_para)
    x1 = torch.rand(4, 3, 2048)
    x2 = torch.rand(4, 3, 2048)
    y1 = encoder(x1)
    y2 = encoder(x2)
    print('Encoded:', y1.size(), y2.size())

    tail = Tail()
    tail_para = utils.get_total_parameters(tail)
    print('Tail:', tail_para)
    z = tail(y1)
    print('Tail:', z.size())

    backbone = Backbone(k=20)
    backbone_para = utils.get_total_parameters(backbone.conv)
    print('Backbone:', backbone_para)
    matrix = backbone(x1, x2)
    print('Reconstruction:', matrix.size())
    features = backbone(x1)
    print('Features:', features.size())


if __name__ == '__main__':
    main()
