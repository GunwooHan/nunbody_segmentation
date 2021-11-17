import os
import argparse
import sklearn
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import SmpModel
from datasets import PoseDataset

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_dir', type=bool, default='saved')
parser.add_argument('--datadir', type=str, default='data/val')
parser.add_argument('--archi', type=str, default='Unet')
parser.add_argument('--backbone', type=str, default='timm-mobilenetv3_small_100')
parser.add_argument('--pretrained_weights', type=str, default=None)
parser.add_argument('--color_output', type=bool, default=True)

args = parser.parse_args()

category_names = ['background', 'body','R-hand', 'L-hand','L-foot','R-foot', 'R-thigh', 'L-thigh',  'R-calf','L-calf',  'L-arm', 'R-arm', 'L-forearm','R-forearm','head']

class_color = [[0, 0, 0], [192, 0, 128],[0, 128, 192],[0, 128, 64],[128, 0, 0],\
        [64, 0, 128],[64, 0, 192],[192, 128, 64],[192, 192, 128],[64, 64, 128],\
        [128, 0, 192],[255, 0, 0],[0, 255, 0],[0, 0, 255],[128, 128, 128]]

def create_label_colormap():
    colormap = np.zeros((15, 3), dtype=np.uint8)
    for inex, (r, g, b) in enumerate(class_color):
        colormap[inex] = [b, g, r]
    return colormap

def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_label_colormap()

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
    
    transform_ = A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2471, 0.2435, 0.2616],
        ),
        ToTensorV2()
    ])
    
    dataset = PoseDataset(data_dir, mode, transform=transform_)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, num_workers=4)    
    
    hist = np.zeros((n_class, n_class))
    
    for idx, data in enumerate(loader):
        img_hist = np.zeros((n_class, n_class))
        img = data[0].cuda()
        output = model(img)
        iou_value = output.argmax(dim=1)
        pred_mask = iou_value[0].detach().cpu().numpy()
        target_mask = data[1].numpy()
    
        if mode in ('train','val') and confusion_matrix:
            if idx == 0 : 
                cf_matrix = sklearn.metrics.confusion_matrix(np.ndarray.flatten(target_mask),
                                              np.ndarray.flatten(pred_mask),
                                              labels=list(range(n_class)))
            else : 
                cf_matrix += sklearn.metrics.confusion_matrix(np.ndarray.flatten(target_mask),
                                              np.ndarray.flatten(pred_mask),
                                              labels=list(range(n_class)))
#         if mIoU_print:

#             img_hist = add_hist(img_hist, target_mask, pred_mask, n_class=n_class)
#             img_hist = add_hist(img_hist, target_mask, pred_mask, n_class=n_class)
#             hist = add_hist(hist, target_mask, pred_mask, n_class=n_class)
#             _, _, img_mIoU, _, _ = label_accuracy_score(img_hist)

#             if img_mIoU > mIoU_best10[-1]: 
#                 mIoU_best10,mIoU_best10_imageid = extract_best_mIoU(index,img_mIoU,mIoU_best10,mIoU_best10_imageid)
#             if img_mIoU < mIoU_worst10[-1]: 
#                 mIoU_worst10,mIoU_worst10_imageid = extract_worst_mIoU(index,img_mIoU,mIoU_worst10,mIoU_worst10_imageid)
                    
    if confusion_matrix:
        draw_confusion_matrix(cf_matrix,'confusion_matrix.png')

    # if mIoU_print:
    #     _, _, mIoU, _, _ = label_accuracy_score(hist)

    # return mIoU, mIoU_best10, mIoU_best10_imageid, mIoU_worst10, mIoU_worst10_imageid


if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = args.save_dir #'saved/Unet_timm-tf_efficientnet_lite4-epoch=10-val/mIoU=0.05-v1.ckpt' 
    model = SmpModel.load_from_checkpoint(model_path,
                                        args=args,
                                        train_transform=None,
                                        val_transform=None)
    
    model = model.cuda()

    main(model, args.datadir, mode = 'train')
    
    
    