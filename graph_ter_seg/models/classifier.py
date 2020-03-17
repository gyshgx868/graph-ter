import torch

import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from graph_ter_seg.tools import utils


class Classifier(nn.Module):
    def __init__(self, in_features=256, num_points=2048, num_classes=16,
                 num_parts=50):
        super(Classifier, self).__init__()
        self.num_points = num_points
        self.conv0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(in_features, 1024, kernel_size=1, bias=False)),
            ('bn0', nn.BatchNorm1d(1024)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2))
        ]))
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(num_classes, 64, kernel_size=1, bias=False)),
            ('bn0', nn.BatchNorm1d(64)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(in_features + 1088, 256, kernel_size=1, bias=False)),
            ('bn0', nn.BatchNorm1d(256)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2)),
            ('drop0', nn.Dropout(p=0.4)),
            ('conv1', nn.Conv1d(256, 256, kernel_size=1, bias=False)),
            ('bn1', nn.BatchNorm1d(256)),
            ('relu1', nn.LeakyReLU(negative_slope=0.2)),
            ('drop1', nn.Dropout(p=0.4)),
            ('conv2', nn.Conv1d(256, 128, kernel_size=1, bias=False)),
            ('bn2', nn.BatchNorm1d(128)),
            ('relu2', nn.LeakyReLU(negative_slope=0.2)),
            ('conv3', nn.Conv1d(128, num_parts, kernel_size=1, bias=False)),
        ]))

    def forward(self, x, labels):
        features = self.conv0(x)
        features = self.max_pool(features)
        labels = self.conv1(labels)
        features = torch.cat((features, labels), dim=1)
        features = features.repeat(1, 1, self.num_points)
        features = torch.cat((features, x), dim=1)
        features = self.classifier(features)
        features = features.permute(0, 2, 1)
        features = F.log_softmax(features, dim=-1)
        return features


def main():
    labels = torch.rand(4, 16, 1)
    features = torch.rand(4, 256, 2048)
    classifier = Classifier()
    print('Classifier:', utils.get_total_parameters(classifier))
    score = classifier(features, labels)
    print('Classification:', score.size())


if __name__ == '__main__':
    main()
