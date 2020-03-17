import torch

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from torch.utils.data.dataloader import DataLoader

from graph_ter_cls.models.backbone import Backbone
from graph_ter_cls.models.layers import Pooler
from graph_ter_cls.runner.runner import Runner
from graph_ter_cls.tools.utils import import_class


class SVMClassifierRunner(Runner):
    def __init__(self, args):
        super(SVMClassifierRunner, self).__init__(args)

    def load_dataset(self):
        feeder_class = import_class(self.args.dataset)
        self.train_feeder = feeder_class(
            self.args.data_path, num_points=self.args.num_points,
            transform=None, phase='train'
        )
        train_data = DataLoader(
            dataset=self.train_feeder,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=8
        )
        self.dataset['train'] = train_data
        self.shape_names = self.train_feeder.shape_names
        self.print_log(f'Train data loaded: {len(self.train_feeder)} samples.')

        if self.args.eval_model:
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
            self.print_log(f'Test data loaded: {len(feeder)} samples.')

    def load_model(self):
        pooler = Pooler()
        pooler = pooler.to(self.output_dev)
        self.model['train'] = pooler
        backbone = Backbone(out_features=self.transform.out_features)
        backbone = backbone.to(self.output_dev)
        self.model['test'] = backbone
        self.classifier = OneVsRestClassifier(
            # 88.1 %
            LinearSVC(C=200.0, max_iter=10240, dual=False, intercept_scaling=12)
        )

    def initialize_model(self):
        if self.args.backbone is not None:
            self.load_model_weights(
                self.model['test'],
                self.args.backbone,
                self.args.ignore_backbone
            )
        else:
            raise ValueError('Please appoint --backbone.')
        self.epoch = 0
        if self.args.classifier is not None:
            self.load_model_weights(
                self.model['train'],
                self.args.classifier,
                self.args.ignore_classifier
            )
            if self.optimizer is not None:
                self.load_optimizer_weights(self.optimizer, self.args.classifier)
            if self.scheduler is not None:
                self.load_scheduler_weights(self.scheduler, self.args.classifier)

    def run(self):
        best_epoch = -1
        best_acc = 0.0
        for epoch in range(self.epoch, self.args.num_epochs):
            self._train_classifier(epoch)
            eval_model = self.args.eval_model and (
                    ((epoch + 1) % self.args.eval_interval == 0) or
                    (epoch + 1 == self.args.num_classifier_epochs))
            if eval_model:
                acc = self._eval_classifier(epoch)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                self.print_log(
                    'Best accuracy: {:.2f}%, best model: model{}.pt'.format(
                        best_acc * 100.0, best_epoch + 1
                    ))

    def _train_classifier(self, epoch):
        self.print_log('Train Classifier Epoch: {}'.format(epoch + 1))
        self.model['test'].eval()
        self.model['train'].eval()

        loader = self.dataset['train']

        self.record_time()
        batch_features = []
        batch_labels = []
        timer = dict(data=0.0, model=0.0, statistic=0.0)
        with torch.no_grad():
            for batch_id, (x, label) in enumerate(loader):
                # get data
                x = x.float().to(self.output_dev)
                label = label.long().to(self.output_dev)
                # forward
                features = self.model['test'](x)
                features = self.model['train'](features)
                # statistic
                if (batch_id + 1) % self.args.log_interval == 0:
                    self.print_log('Batch({}/{}) done.'.format(
                        batch_id + 1, len(loader)
                    ))
                batch_features.append(features.cpu().numpy())
                batch_labels.append(label.cpu().numpy().reshape((-1, 1)))

        timer['data'] += self.tick()
        batch_features = np.vstack(batch_features)
        batch_labels = np.vstack(batch_labels).reshape((-1,)).astype(int)
        self.classifier.fit(batch_features, batch_labels)
        timer['model'] += self.tick()

        self.print_log(
            'Time consumption: [Data] {:.1f} min, [Model] {:.1f} min'.format(
                timer['data'] / 60.0, timer['model'] / 60.0
            ))

    def _eval_classifier(self, epoch):
        self.print_log('Eval Classifier Epoch: {}'.format(epoch + 1))
        self.model['train'].eval()
        self.model['test'].eval()

        loader = self.dataset['test']
        pred_scores = []
        true_scores = []
        with torch.no_grad():
            for batch_id, (x, label) in enumerate(loader):
                # get data
                x = x.float().to(self.output_dev)
                label = label.long().to(self.output_dev)

                # forward
                features = self.model['test'](x)
                features = self.model['train'](features)
                current_features = features.detach().cpu().numpy()
                current_labels = label.cpu().numpy()

                # statistic
                if (batch_id + 1) % self.args.log_interval == 0:
                    self.print_log('Batch({}/{}) done.'.format(
                        batch_id + 1, len(loader)
                    ))
                pred = self.classifier.predict(current_features)
                pred_scores.append(pred)
                true_scores.append(current_labels)
        pred_scores = np.concatenate(pred_scores)
        true_scores = np.concatenate(true_scores)

        overall_acc = accuracy_score(true_scores, pred_scores)
        avg_class_acc = balanced_accuracy_score(true_scores, pred_scores)
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

        return overall_acc
