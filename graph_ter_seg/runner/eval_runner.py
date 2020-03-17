import os
import torch

import numpy as np
import pandas as pd
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader

from graph_ter_seg.models.backbone import Backbone
from graph_ter_seg.models.classifier import Classifier
from graph_ter_seg.runner.runner import Runner
from graph_ter_seg.tools.utils import import_class


class EvaluationRunner(Runner):
    def __init__(self, args):
        super(EvaluationRunner, self).__init__(args)
        # loss
        self.loss = nn.NLLLoss().to(self.output_dev)

    def load_dataset(self):
        feeder_class = import_class(self.args.dataset)
        self.feeder = feeder_class(
            self.args.data_path, num_points=self.args.num_points,
            transform=None, phase='test'
        )
        test_data = DataLoader(
            dataset=self.feeder,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=8
        )
        self.dataset['test'] = test_data

        self.classes_dict = self.feeder.classes_dict
        self.shape_names = self.feeder.shape_names
        self.num_classes = self.feeder.num_classes
        self.num_parts = self.feeder.num_parts

        self.print_log(f'Test data loaded: {len(self.feeder)} samples.')

    def load_model(self):
        backbone = Backbone(
            k=self.args.knn, out_features=self.transform.out_features
        )
        backbone = backbone.to(self.output_dev)
        self.model['train'] = backbone
        classifier = Classifier(
            num_points=self.args.num_points, num_classes=self.num_classes,
            num_parts=self.num_parts
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
        accuracy_values = []
        iou_values = []
        iou_table = np.zeros(shape=(len(self.shape_names), 3))
        if self.args.save_segmentation:
            shape_count = dict()
            unique_shape_names = []
            for key in self.classes_dict.keys():
                shape_count.update({key: 0})
                unique_shape_names.append(key)
        with torch.no_grad():
            for batch_id, (x, labels, target) in enumerate(loader):
                # get data
                x = x.float().to(self.output_dev)
                labels = labels.float().to(self.output_dev)
                target = target.long().to(self.output_dev)

                # forward
                features = self.model['train'](x)
                pred = self.model['test'](features, labels)

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

                # save segmentation results for each batch
                if self.args.save_segmentation:
                    batch_target = np.reshape(
                        target.cpu().numpy(), newshape=(batch_size, -1)
                    )
                    batch_pred = np.reshape(
                        pred_indices.cpu().numpy(), newshape=(batch_size, -1)
                    )
                    batch_label = np.reshape(
                        labels.data.max(1)[1].cpu().numpy(), newshape=(-1,)
                    )
                    batch_points = x.cpu().numpy()
                    for i in range(batch_size):
                        shape_name = unique_shape_names[batch_label[i]]
                        file_name = shape_name + str(shape_count[shape_name])
                        shape_count[shape_name] += 1
                        gt_name = os.path.join(
                            self.cloud_path, file_name + '_gt.obj'
                        )
                        pred_name = os.path.join(
                            self.cloud_path, file_name + '.obj'
                        )
                        self.feeder.output_colored_point_cloud(
                            points=batch_points[i, :, :],
                            seg=batch_target[i, :],
                            output_file=gt_name
                        )
                        self.feeder.output_colored_point_cloud(
                            points=batch_points[i, :, :],
                            seg=batch_pred[i, :],
                            output_file=pred_name
                        )

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
