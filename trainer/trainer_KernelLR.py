import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_valid = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 1 #int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', 'cls_loss', 'reg_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', 'cls_loss', 'reg_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, K = self.model.fit(data)
            loss, loss_cls, loss_reg = self.model.cal_loss(output, target, K)            
            loss.backward()
            self.optimizer.step()
            
            # BUG: loss != cls + reg
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('cls_loss', loss_cls.item())
            self.train_metrics.update('reg_loss', loss_reg.item())
            
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
            
            # print(loss.item(), loss_cls.item(), loss_reg.item())

            if (batch_idx+1) % self.log_step == 0:
                logstr = 'Train Epoch: {} {} Loss: {:.6f} [Cls_Loss: {:.6f} / Reg_Loss: {:.6f}]'.format(
                    epoch, self._progress(batch_idx),
                    self.train_metrics.current('loss')/self.log_step,
                    self.train_metrics.current('cls_loss')/self.log_step,
                    self.train_metrics.current('reg_loss')/self.log_step)

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                # log loss
                # self.train_metrics.log('loss', log_step=self.log_step)
                # log metrics
                for met in self.metric_ftns:                    
                    logstr = logstr + \
                        " {}: {:.3f}".format(
                            met.__name__, self.train_metrics.current(met.__name__)/self.log_step)
                    # self.train_metrics.log(met.__name__, log_step=self.log_step)

                self.train_metrics.log_all(log_step=self.log_step)
                
                self.logger.debug(logstr)             
                

                # visulization
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_valid:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about valid
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output, _ = self.model.predict(data)
                loss, cls_loss, reg_loss = self.model.cal_loss(output, target)

                # update record
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('cls_loss', cls_loss.item())
                self.valid_metrics.update('reg_loss', reg_loss.item())
                
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
        self.valid_metrics.log_all()
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
