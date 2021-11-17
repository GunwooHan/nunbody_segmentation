import os
import random
import argparse
import cv2
import numpy as np
import pandas as pd
import webcolors
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pycocotools.coco import COCO
import segmentation_models_pytorch as smp
from torchvision import models
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from torchinfo import summary

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pytorch_lightning as pl
from torchvision import models
from transform import make_transform
from model import SmpModel
from datasets import PoseDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_datadir', type=str, default='data/train')
parser.add_argument('--val_datadir', type=str, default='data/val')

parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--archi', type=str, default='Unet')
parser.add_argument('--backbone', type=str, default='efficientnet-b6')
parser.add_argument('--pretrained_weights', type=str, default='imagenet')
parser.add_argument('--fp16', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
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

class_colormap = pd.DataFrame({'name': ['background',
                'body',
                'R-hand',
                'L-hand',
                'L-foot',
                'R-foot',
                'R-thigh',
                'L-thigh',
                'R-calf',
                'L-calf',
                'L-arm',
                'R-arm',
                'L-forearm',
                'R-forearm',
                'head'],
                 'r': [0, 192, 0  , 0  , 128, 64 , 64 , 192, 192, 64 , 128, 0  , 0  , 64 , 192],
                 'g': [0, 0  , 128, 128, 0  , 0  , 0  , 128, 192, 64 , 0  , 0  , 128, 128, 192],
                 'b': [0, 128, 192, 64 , 0  , 128, 192, 64 , 128, 128, 192, 128, 0  , 192, 192]})

category_names = ['background', 'body','R-hand', 'L-hand','L-foot','R-foot', 'R-thigh', 'L-thigh',  'R-calf','L-calf',  'L-arm', 'R-arm', 'L-forearm','R-forearm','head']

def collate_fn(batch):
    return tuple(zip(*batch))

########################### DataLoader define ###########################
    
def create_trash_label_colormap():
    """Creates a label colormap used in Trash segmentation.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((15, 3), dtype=np.uint8)
    for inex, (_, r, g, b) in enumerate(class_colormap.values):
        colormap[inex] = [r, g, b]
    
    return colormap

def label_to_color_image(label):

    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_trash_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

########################### mIoU 계산 ###########################
    
def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, iu

def add_hist(hist, label_trues, label_preds, n_class):
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
        
def draw_confusion_matrix(cf_matrix, save_dir = 'confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(10,10))
    a=cf_matrix.astype('float')/cf_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(a, index = category_names, columns = category_names)
    sns.heatmap(df_cm.round(2), annot = True, cmap = plt.cm.Blues)
    plt.savefig(save_dir)
    
########################### mIoU추출 ###########################        

def extract_best_mIoU(index, now_mIoU, mIoU_list, best_img_list):
    temp_miou = []
    temp_img = []
    skip = True
    for i in range(10):
        if now_mIoU >= mIoU_list[i] and skip:
            temp_miou.append(now_mIoU)
            temp_img.append(index)
            skip = False
        temp_miou.append(mIoU_list[i])
        temp_img.append(best_img_list[i])
    return temp_miou[:10], temp_img[:10]


def extract_worst_mIoU(index, now_mIoU, mIoU_list, best_img_list):
    temp_miou = []
    temp_img = []
    skip = True
    for i in range(10):
        if now_mIoU <= mIoU_list[i] and skip:
            temp_miou.append(now_mIoU)
            temp_img.append(index)
            skip = False
        temp_miou.append(mIoU_list[i])
        temp_img.append(best_img_list[i])
    return temp_miou[:10], temp_img[:10]

########################### Main 함수 ###########################

def main(model, 
         data_dir, 
         mode = 'train',
         n_class = 15,
         confusion_matrix= True, 
         save_image_pred = True, 
         mIoU_print = True):
    if mode not in ('train','val'):
        raise print('using train or val dataset')
    
    mIoU_best10 = [0,0,0,0,0,0,0,0,0,0]
    mIoU_worst10 = [99,99,99,99,99,99,99,99,99,99]
    mIoU_best10_imageid = [0,0,0,0,0,0,0,0,0,0]
    mIoU_worst10_imageid = [0,0,0,0,0,0,0,0,0,0]
    
    dataset = PoseDataset(data_dir=data_dir, mode=mode, transform=A.Compose([ToTensorV2()]))
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size = 1, num_workers=4, collate_fn=collate_fn)    
    
    hist = np.zeros((n_class, n_class))
    with torch.no_grad():
        for index, data in tqdm(enumerate(loader)):
            img_hist = np.zeros((n_class, n_class))
            img = data[0][0].cuda()
            # model.eval()
            
            outs = model(img)
            oms = outs.argmax(dim=1).detach().cpu().numpy()
            masks = data[1][0].numpy()
            
            if mode in ('train','val') and confusion_matrix:
                if index == 0 : 
                    cf_matrix = sklearn.metrics.confusion_matrix(np.ndarray.flatten(masks),
                                                  np.ndarray.flatten(oms[0]),
                                                  labels=list(range(n_class)))
                else : 
                    cf_matrix += sklearn.metrics.confusion_matrix(np.ndarray.flatten(masks),
                                                  np.ndarray.flatten(oms[0]),
                                                  labels=list(range(n_class)))
                    
            if mIoU_print:
                
                img_hist = add_hist(img_hist, masks, oms, n_class=n_class)
                img_hist = add_hist(img_hist, masks, oms, n_class=n_class)
                hist = add_hist(hist, masks, oms, n_class=n_class)
                _, _, img_mIoU, _, _ = label_accuracy_score(img_hist)
                
                if img_mIoU > mIoU_best10[-1]: 
                    mIoU_best10,mIoU_best10_imageid = extract_best_mIoU(index,img_mIoU,mIoU_best10,mIoU_best10_imageid)
                if img_mIoU < mIoU_worst10[-1]: 
                    mIoU_worst10,mIoU_worst10_imageid = extract_worst_mIoU(index,img_mIoU,mIoU_worst10,mIoU_worst10_imageid)
                    
        if confusion_matrix:
            draw_confusion_matrix(cf_matrix,'confusion_matrix.png')

        if mIoU_print:
            _, _, mIoU, _, _ = label_accuracy_score(hist)
    
    return mIoU, mIoU_best10, mIoU_best10_imageid, mIoU_worst10, mIoU_worst10_imageid

if __name__ == '__main__':
    model_path = 'saved/Unet_timm-tf_efficientnet_lite4-epoch=10-val/mIoU=0.05-v1.ckpt' 
    model = SmpModel.load_from_checkpoint(model_path,
                                        args=args,
                                        train_transform=None,
                                        val_transform=None)
    
    mIoU, mIoU_best10, mIoU_best10_imageid, mIoU_worst10, mIoU_worst10_imageid = main(model, args.val_datadir, mode = 'train')
    
    