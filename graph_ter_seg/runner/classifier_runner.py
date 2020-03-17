import os
import torch

import numpy as np
import pandas as pd
import torch.nn as nn

from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from graph_ter_seg.models.backbone import Backbone
from graph_ter_seg.models.classifier import Classifier
from graph_ter_seg.runner.runner import Runner
from graph_ter_seg.tools.utils import import_class


class ClassifierRunner(Runner):
    def __init__(self, args):
        super(ClassifierRunner, self).__init__(args)
        # loss
        self.loss = nn.NLLLoss().to(self.output_dev)

    def load_dataset(self):
        feeder_class = import_class(self.args.dataset)
        feeder = feeder_class(
            self.args.data_path, num_points=self.args.num_points,
            transform=None, phase='train'
        )
        train_data = DataLoader(
            dataset=feeder,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=8
        )
        self.dataset['train'] = train_data

        self.shape_names = feeder.shape_names
        self.num_classes = feeder.num_classes
        self.num_parts = feeder.num_parts
        self.print_log(f'Train data loaded: {len(feeder)} samples.')

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
        classifier = Classifier(
            num_points=self.args.num_points, num_classes=self.num_classes,
            num_parts=self.num_parts
        )
        classifier = classifier.to(self.output_dev)
        self.model['train'] = classifier
        backbone = Backbone(
            k=self.args.knn, out_features=self.transform.out_features
        )
        backbone = backbone.to(self.output_dev)
        self.model['test'] = backbone

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
            self.load_optimizer_weights(self.optimizer, self.args.classifier)
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
                    'Best IoU: {:.2f}%, best model: model{}.pt'.format(
                        best_acc * 100.0, best_epoch + 1
                    ))

    def _train_classifier(self, epoch):
        self.print_log(f'Train Classifier Epoch: {epoch + 1}')
        self.model['test'].eval()
        self.model['train'].train()

        loader = self.dataset['train']
        loss_values = []

        self.record_time()
        timer = dict(data=0.0, model=0.0, statistic=0.0)
        for batch_id, (x, labels, target) in enumerate(loader):
            # get data
            x = x.float().to(self.output_dev)
            labels = labels.float().to(self.output_dev)
            target = target.long().to(self.output_dev)
            target = target.view(-1, 1)[:, 0]
            timer['data'] += self.tick()

            # forward
            features = self.model['test'](x)
            pred = self.model['train'](features, labels)
            pred = pred.contiguous().view(-1, self.num_parts)
            loss = self.loss(pred, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            timer['model'] += self.tick()

            # statistic
            loss_values.append(loss.item())
            if (batch_id + 1) % self.args.log_interval == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}, lr: {:.5f}'.format(
                        batch_id + 1, len(loader), loss.item(),
                        self.optimizer.param_groups[0]['lr']
                    ))
            timer['statistic'] += self.tick()
        self.scheduler.step()

        mean_loss = np.mean(loss_values)
        self.print_log('Mean training loss: {:.4f}.'.format(mean_loss))
        self.print_log(
            'Time consumption: [Data] {:.1f} min, [Model] {:.1f} min'.format(
                timer['data'] / 60.0, timer['model'] / 60.0
            ))

        if self.args.save_model and (epoch + 1) % self.args.save_interval == 0:
            model_path = os.path.join(
                self.classifier_path, f'model{epoch + 1}.pt'
            )
            self.save_weights(
                epoch, self.model['train'], self.optimizer,
                self.scheduler, model_path
            )

        if self.args.use_tensorboard:
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('train/classifier_loss', mean_loss, epoch)

    def _eval_classifier(self, epoch):
        self.print_log(f'Eval Classifier Epoch: {epoch + 1}')
        self.model['train'].eval()
        self.model['test'].eval()

        loader = self.dataset['test']
        loss_values = []
        accuracy_values = []
        iou_values = []
        iou_table = np.zeros(shape=(len(self.shape_names), 3))
        with torch.no_grad():
            for batch_id, (x, labels, target) in enumerate(loader):
                # get data
                x = x.float().to(self.output_dev)
                labels = labels.float().to(self.output_dev)
                target = target.long().to(self.output_dev)

                # forward
                features = self.model['test'](x)
                pred = self.model['train'](features, labels)

                # statistic
                iou_table, iou = self._compute_cat_iou(pred, target, iou_table)
                iou_values += iou
                pred = pred.contiguous().view(-1, self.num_parts)
                target = target.view(-1, 1)[:, 0]
                loss = self.loss(pred, target)
                loss_values.append(loss.item())
                pred_indices = pred.data.max(1)[1]
                corrected = pred_indices.eq(target.data).cpu().sum()
                batch_size, _, num_points = x.size()
                accuracy_values.append(
                    corrected.item() / (batch_size * num_points))
                if (batch_id + 1) % self.args.log_interval == 0:
                    self.print_log(
                        'Batch({}/{}) done. Loss: {:.4f}'.format(
                            batch_id + 1, len(loader), loss.item()
                        ))

        mean_loss = np.mean(loss_values)
        mean_accuracy = np.mean(accuracy_values)
        mean_iou = np.mean(iou_values)
        self.print_log('Mean testing loss: {:.4f}.'.format(mean_loss))
        self.print_log('Mean accuracy: {:.2f}%.'.format(mean_accuracy * 100.0))
        self.print_log('Mean IoU: {:.2f}%.'.format(mean_iou * 100.0))

        if self.args.show_details:
            self.print_log('Detailed results:')
            iou_table[:, 2] = iou_table[:, 0] / iou_table[:, 1]
            iou_table = pd.DataFrame(
                iou_table, columns=['IoU', 'Count', 'Mean']
            )
            iou_table['Class'] = [
                self.shape_names[i] for i in range(len(self.shape_names))
            ]
            self.print_log(iou_table.round(4), print_time=False)

        if self.args.use_tensorboard:
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('test/loss', mean_loss, epoch)
                writer.add_scalar('test/mean_accuracy', mean_accuracy, epoch)
                writer.add_scalar('test/mean_iou', mean_iou, epoch)

        return mean_iou

    @staticmethod
    def _compute_cat_iou(pred, target, iou_table):
        iou_list = []
        target = target.cpu().data.numpy()
        for j in range(pred.size(0)):
            batch_pred = pred[j]
            batch_target = target[j]
            batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()
            for cat in np.unique(batch_target):
                intersection = np.sum(
                    np.logical_and(batch_choice == cat, batch_target == cat)
                )
                union = np.sum(
                    np.logical_or(batch_choice == cat, batch_target == cat)
                )
                iou = intersection / float(union) if union != 0 else 1.0
                iou_table[cat, 0] += iou
                iou_table[cat, 1] += 1
                iou_list.append(iou)
        return iou_table, iou_list
