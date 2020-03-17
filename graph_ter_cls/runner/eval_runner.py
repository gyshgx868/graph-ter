import numpy as np
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from torch.utils.data.dataloader import DataLoader

from graph_ter_cls.models.backbone import Backbone
from graph_ter_cls.models.classifier import Classifier
from graph_ter_cls.runner.runner import Runner
from graph_ter_cls.tools.utils import import_class


class EvaluationRunner(Runner):
    def __init__(self, args):
        super(EvaluationRunner, self).__init__(args)
        # loss
        self.loss = nn.CrossEntropyLoss().to(self.output_dev)

    def load_dataset(self):
        feeder_class = import_class(self.args.dataset)
        feeder = feeder_class(
            self.args.data_path, num_points=self.args.num_points,
            transform=None, phase='test'
        )
        test_data = DataLoader(
            dataset=feeder,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=8
        )
        self.dataset['test'] = test_data
        self.num_classes = feeder.num_classes
        self.shape_names = feeder.shape_names
        self.print_log(f'Test data loaded: {len(feeder)} samples.')

    def load_model(self):
        backbone = Backbone(
            k=self.args.knn, out_features=self.transform.out_features
        )
        backbone = backbone.to(self.output_dev)
        self.model['train'] = backbone
        classifier = Classifier(
            dropout=self.args.dropout, num_classes=self.num_classes
        )
        classifier = classifier.to(self.output_dev)
        self.model['test'] = classifier

    def initialize_model(self):
        if self.args.backbone is not None and self.args.classifier is not None:
            self.load_model_weights(
                self.model['train'],
                self.args.backbone,
                self.args.ignore_backbone
            )
            self.load_model_weights(
                self.model['test'],
                self.args.classifier,
                self.args.ignore_classifier
            )
        else:
            raise ValueError('Please appoint --backbone and --classifier.')

    def run(self):
        self._eval_classifier()

    def _eval_classifier(self):
        self.print_log('Eval Classifier:')
        self.model['train'].eval()
        self.model['test'].eval()

        loader = self.dataset['test']
        loss_values = []
        pred_scores = []
        true_scores = []

        for batch_id, (x, label) in enumerate(loader):
            # get data
            x = x.float().to(self.output_dev)
            label = label.long().to(self.output_dev)

            # forward
            features = self.model['train'](x)
            y = self.model['test'](features)
            loss = self.loss(y, label)

            # statistic
            loss_values.append(loss.item())
            if (batch_id + 1) % self.args.log_interval == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}'.format(
                        batch_id + 1, len(loader), loss.item()
                    ))
            pred = y.max(dim=1)[1]
            pred_scores.append(pred.data.cpu().numpy())
            true_scores.append(label.data.cpu().numpy())
        pred_scores = np.concatenate(pred_scores)
        true_scores = np.concatenate(true_scores)

        mean_loss = np.mean(loss_values)
        overall_acc = accuracy_score(true_scores, pred_scores)
        avg_class_acc = balanced_accuracy_score(true_scores, pred_scores)
        self.print_log('Mean testing loss: {:.4f}.'.format(mean_loss))
        self.print_log('Overall accuracy: {:.2f}%'.format(overall_acc * 100.0))
        self.print_log(
            'Average class accuracy: {:.2f}%'.format(avg_class_acc * 100.0)
        )

        if self.args.show_details:
            self.print_log('Detailed results:')
            report = classification_report(
                true_scores,
                pred_scores,
                target_names=self.shape_names,
                digits=4
            )
            self.print_log(report, print_time=False)
