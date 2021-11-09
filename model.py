import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduelr
# from adamp import AdamP
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.functional import iou, accuracy
import albumentations as A

from datasets import PoseDataset


class SmpModel(pl.LightningModule):
    def __init__(self, args=None, train_transform=None, val_transform=None):
        super().__init__()
        arhci_name_list = sorted([name for name in smp.__dict__ if not (name.islower() or name.startswith('__'))])

        assert (args.archi in arhci_name_list), \
            (f"[!] Architecture Name is wrong, check Archi config, expected: {arhci_name_list} received: {args.archi}")

        self.model = getattr(smp, args.archi)(
            encoder_name=args.backbone,
            encoder_weights=args.pretrained_weights,
            in_channels=3,
            classes=15,
        )

        if train_transform and val_transform:
            self.batch_size = args.batch_size
            self.train_datadir = args.train_datadir
            self.val_datadir = args.val_datadir
            self.train_transform = train_transform
            self.val_transform = val_transform

            self.args = args

            if args.loss == 'ce':
                self.criterion = nn.CrossEntropyLoss()

            self.train_data = PoseDataset(self.train_datadir, mode='train', transform=self.train_transform)
            self.val_data = PoseDataset(self.val_datadir, mode='val', transform=self.val_transform)

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        # elif self.args.optimizer == 'adamp':
        #     optimizer = AdamP(self.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)

        if self.args.scheduler == "reducelr":
            scheduler = lr_scheduelr.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode="max", verbose=True)
        elif self.args.scheduler == "cosineanneal":
            scheduler = lr_scheduelr.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min= 1e-5,
                                                    last_epoch=-1, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/mIoU"}

    def training_step(self, train_batch, batch_idx):
        image, mask = train_batch
        outputs = self.model(image)
        loss = self.criterion(outputs, mask)
        iou_value = iou(outputs.argmax(dim=1), mask)
        acc_value = accuracy(outputs, mask)

        self.log('train/loss', loss)
        self.log('train/acc', acc_value)
        self.log('train/mIoU', iou_value)

        return {"loss": loss, "IoU": iou_value, "acc": acc_value}

    def validation_step(self, val_batch, batch_idx):
        image, mask = val_batch
        outputs = self.model(image)
        loss = self.criterion(outputs, mask)
        iou_value = iou(outputs.argmax(dim=1), mask)
        acc_value = accuracy(outputs, mask)

        self.log('val/loss', loss)
        self.log('val/acc', acc_value)
        self.log('val/mIoU', iou_value)

        return {"loss": loss, "IoU": iou_value, "acc": acc_value}

    def training_epoch_end(self, outputs):
        total_loss = 0.0
        total_iou = 0.0
        total_acc = 0.0

        iter_count = len(outputs)

        for idx in range(iter_count):
            total_loss += outputs[idx]['loss'].item()
            total_iou += outputs[idx]['IoU'].item()
            total_acc += outputs[idx]['acc'].item()

        self.log('train/epoch_loss', total_loss / iter_count)
        self.log('train/epoch_acc', total_acc / iter_count)
        self.log('train/epoch_mIoU', total_iou / iter_count)

    def validation_epoch_end(self, outputs):
        total_loss = 0.0
        total_iou = 0.0
        total_acc = 0.0

        iter_count = len(outputs)

        for idx in range(iter_count):
            total_loss += outputs[idx]['loss'].item()
            total_iou += outputs[idx]['IoU'].item()
            total_acc += outputs[idx]['acc'].item()

        self.log('val/epoch_loss', total_loss / iter_count)
        self.log('val/epoch_acc', total_acc / iter_count)
        self.log('val/epoch_mIoU', total_iou / iter_count)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.args.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=1, num_workers=self.args.num_workers)