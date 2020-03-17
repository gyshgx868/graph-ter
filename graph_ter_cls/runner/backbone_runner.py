import os
import torch

import numpy as np
import torch.nn as nn

from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from graph_ter_cls.models.backbone import Backbone
from graph_ter_cls.runner.runner import Runner
from graph_ter_cls.tools.utils import import_class


class BackboneRunner(Runner):
    def __init__(self, args):
        super(BackboneRunner, self).__init__(args)
        # loss
        self.loss = nn.MSELoss().to(self.output_dev)

    def load_dataset(self):
        feeder_class = import_class(self.args.dataset)
        feeder = feeder_class(
            self.args.data_path, num_points=self.args.num_points,
            transform=self.transform, phase='train'
        )
        self.dataset['train'] = DataLoader(
            dataset=feeder,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=8
        )
        self.print_log(f'Train data loaded: {len(feeder)} samples.')

    def load_model(self):
        model = Backbone(
            k=self.args.knn, out_features=self.transform.out_features
        )
        model = model.to(self.output_dev)
        self.model['train'] = model

    def initialize_model(self):
        if self.args.backbone is not None:
            self.load_model_weights(
                self.model['train'],
                self.args.backbone,
                self.args.ignore_backbone
            )
            self.load_optimizer_weights(self.optimizer, self.args.backbone)
            self.load_scheduler_weights(self.scheduler, self.args.backbone)

    def run(self):
        best_epoch = -1
        best_loss = np.Inf
        for epoch in range(self.epoch, self.args.num_epochs):
            loss = self._train_backbone(epoch)
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
            self.print_log(
                'Min loss: {:.5f}, best model: model{}.pt'.format(
                    best_loss, best_epoch + 1
                ))

    def _train_backbone(self, epoch):
        self.print_log(f'Train Backbone Epoch: {epoch + 1}')
        self.model['train'].train()

        loss_values = []

        self.record_time()
        timer = dict(data=0.0, model=0.0, statistic=0.0)
        for batch_id, (x, y, t, m, _) in enumerate(self.dataset['train']):
            # get data
            x = x.float().to(self.output_dev)
            y = y.float().to(self.output_dev)
            t = t.float().to(self.output_dev)
            m = m.long().to(self.output_dev)
            timer['data'] += self.tick()

            # forward
            t_hat = self.model['train'](x, y)
            t_hat = torch.gather(t_hat, dim=-1, index=m)
            loss = self.loss(t, t_hat) * self.args.lambda_mse

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            timer['model'] += self.tick()

            loss_values.append(loss.item())
            if (batch_id + 1) % self.args.log_interval == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}, lr: {:.5f}'.format(
                        batch_id + 1, len(self.dataset['train']),
                        loss.item(), self.optimizer.param_groups[0]['lr']
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
                self.backbone_path, 'model{}.pt'.format(epoch + 1)
            )
            self.save_weights(
                epoch, self.model['train'], self.optimizer, self.scheduler,
                model_path
            )

        if self.args.use_tensorboard:
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('train/backbone_loss', mean_loss, epoch)
        return mean_loss
