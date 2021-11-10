import os
import argparse

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transform import make_transform
from model import SmpModel

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_datadir', type=str, default='data/train')
parser.add_argument('--val_datadir', type=str, default='data/val')

parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--archi', type=str, default='Unet')
parser.add_argument('--backbone', type=str, default='efficientnet-b6')
parser.add_argument('--pretrained_weights', type=str, default='imagenet')
parser.add_argument('--fp16', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--auto_batch_size', type=bool, default=False)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--scheduler', type=str, default='reducelr')
parser.add_argument('--loss', type=str, default='ce')

parser.add_argument('--RandomBrightnessContrast', type=float, default=0)
parser.add_argument('--HueSaturationValue', type=float, default=0)
parser.add_argument('--RGBShift', type=float, default=0)
parser.add_argument('--RandomGamma', type=float, default=0)
parser.add_argument('--HorizontalFlip', type=float, default=0)
parser.add_argument('--VerticalFlip', type=float, default=0)
parser.add_argument('--ImageCompression', type=float, default=0)
parser.add_argument('--ShiftScaleRotate', type=float, default=0)
parser.add_argument('--ShiftScaleRotateMode', type=int, default=4) # Constant, Replicate, Reflect, Wrap, Reflect101
parser.add_argument('--Downscale', type=float, default=0)
parser.add_argument('--GridDistortion', type=float, default=0)
parser.add_argument('--MotionBlur', type=float, default=0)
parser.add_argument('--RandomResizedCrop', type=float, default=0)
parser.add_argument('--CLAHE', type=float, default=0)

args = parser.parse_args()

if __name__ == '__main__':
    # SWA = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=0.001, annealing_epochs=5, annealing_strategy='cos')
    pl.seed_everything(args.seed)
    wandb_logger = WandbLogger(project='NunBody', name=f'{args.backbone}_{args.archi}')
    wandb_logger.log_hyperparams(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/mIoU",
        dirpath="saved",
        filename=f"{args.archi}_{args.backbone}"+"-{epoch:02d}-{val/mIoU:.2f}",
        save_top_k=1,
        mode="max",
        save_weights_only=True
    )
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=5, verbose=False, mode="min")

    train_transform, val_transform = make_transform(args)

    model = SmpModel(args, train_transform, val_transform)

    trainer = pl.Trainer(gpus=args.gpus,
                         precision=args.fp16,
                         max_epochs=args.epochs,
                         # log_every_n_steps=1,
                         accelerator='ddp',
                         # num_sanity_val_steps=0,
                         # limit_train_batches=50,
                         # limit_val_batches=10,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, early_stop_callback])  # , callbacks=[SWA])

    trainer.fit(model)

