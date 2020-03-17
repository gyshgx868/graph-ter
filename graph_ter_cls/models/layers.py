import torch
import torch.nn as nn

from graph_ter_cls.tools import utils


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


class Pooler(nn.Module):
    def __init__(self):
        super(Pooler, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.max_pool(x).view(batch_size, -1)
        x2 = self.avg_pool(x).view(batch_size, -1)
        x = torch.cat((x1, x2), dim=1)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def main():
    layer = EdgeConvolution(k=10, in_features=3, out_features=64)
    print('Parameters:', utils.get_total_parameters(layer))
    x = torch.rand(3, 3, 1024)
    y = layer(x)
    print(y.size())


if __name__ == '__main__':
    main()
