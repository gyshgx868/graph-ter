import torch.nn as nn

from collections import OrderedDict

from graph_ter_seg.tools import utils


class EdgeConvolution(nn.Module):
    def __init__(self, k, in_features, out_features):
        super(EdgeConvolution, self).__init__()
        self.k = k
        self.conv = nn.Conv2d(
            in_features * 2, out_features, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = utils.get_edge_feature(x, k=self.k)
        x = self.relu(self.bn(self.conv(x)))
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class MultiEdgeConvolution(nn.Module):
    def __init__(self, k, in_features, mlp):
        super(MultiEdgeConvolution, self).__init__()
        self.k = k
        self.conv = nn.Sequential()
        for index, feature in enumerate(mlp):
            if index == 0:
                layer = nn.Sequential(OrderedDict([
                    ('conv%d' %index, nn.Conv2d(
                        in_features * 2, feature, kernel_size=1, bias=False
                    )),
                    ('bn%d' % index, nn.BatchNorm2d(feature)),
                    ('relu%d' % index, nn.LeakyReLU(negative_slope=0.2))
                ]))
            else:
                layer = nn.Sequential(OrderedDict([
                    ('conv%d' %index, nn.Conv2d(
                        mlp[index - 1], feature, kernel_size=1, bias=False
                    )),
                    ('bn%d' % index, nn.BatchNorm2d(feature)),
                    ('relu%d' % index, nn.LeakyReLU(negative_slope=0.2))
                ]))
            self.conv.add_module('layer%d' % index, layer)

    def forward(self, x):
        x = utils.get_edge_feature(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def main():
    conv = MultiEdgeConvolution(k=20, mlp=(64, 64), in_features=64)
    print(conv)


if __name__ == '__main__':
    main()
