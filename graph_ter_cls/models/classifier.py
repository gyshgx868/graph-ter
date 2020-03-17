import torch
import torch.nn as nn

from collections import OrderedDict

from graph_ter_cls.models.layers import Pooler
from graph_ter_cls.tools import utils


class Classifier(nn.Module):
    def __init__(self, dropout=0.5, in_features=1024, num_classes=40):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(in_features, 1024, kernel_size=1, bias=False)),
            ('bn0', nn.BatchNorm1d(1024)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2)),
            ('conv1', nn.Conv1d(1024, 768, kernel_size=1, bias=False)),
            ('bn1', nn.BatchNorm1d(768)),
            ('relu1', nn.LeakyReLU(negative_slope=0.2)),
            ('conv2', nn.Conv1d(768, 512, kernel_size=1, bias=False)),
            ('bn2', nn.BatchNorm1d(512)),
            ('relu2', nn.LeakyReLU(negative_slope=0.2))
        ]))
        self.pool = Pooler()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 512, bias=False)),
            ('bn1', nn.BatchNorm1d(512)),
            ('relu1', nn.LeakyReLU(negative_slope=0.2)),
            ('drop1', nn.Dropout(p=dropout)),
            ('fc2', nn.Linear(512, 256)),
            ('bn2', nn.BatchNorm1d(256)),
            ('relu2', nn.LeakyReLU(negative_slope=0.2)),
            ('drop2', nn.Dropout(p=dropout)),
            ('fc3', nn.Linear(256, num_classes))
        ]))

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def main():
    features = torch.rand(4, 1024, 1024)
    classifier = Classifier()
    print('Classifier:', utils.get_total_parameters(classifier))
    score = classifier(features)
    print('Classification:', score.size())


if __name__ == '__main__':
    main()
