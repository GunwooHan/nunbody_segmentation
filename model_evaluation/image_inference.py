import os
import pickle
import cv2
import argparse
import matplotlib.pyplot as plt
import albumentations as A
from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import webcolors
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from model import SmpModel
from datasets import PoseDataset


parser = argparse.ArgumentParser()

parser.add_argument('--datadir', type=str, default='test')
parser.add_argument('--save_path', type=str, default='val_check')
parser.add_argument('--color_output', type=bool, default=True)

args = parser.parse_args()


class_color = [[0, 0, 0], [192, 0, 128],[0, 128, 192],[0, 128, 64],[128, 0, 0],\
        [64, 0, 128],[64, 0, 192],[192, 128, 64],[192, 192, 128],[64, 64, 128],\
        [128, 0, 192],[255, 0, 0],[0, 255, 0],[0, 0, 255],[128, 128, 128]]
category_names = ['background', 'body','R-hand', 'L-hand','L-foot','R-foot', 'R-thigh', 'L-thigh',  'R-calf','L-calf',  'L-arm', 'R-arm', 'L-forearm','R-forearm','head']

def parse_args():
    parser = argparse.ArgumentParser(description='Inference and make the csv file for submission to AI-Stage')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('save_path', default = 'inference_image', help='output file path to save results')
    
    return parser.parse_args()


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = np.array(class_color)
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]

def make_legend_elements(category_names, class_color):
    category_and_rgb = [[category_names[idx], (r,g,b)] for idx, ( r, g, b) in enumerate(class_color)]
    legend_elements = [Patch(facecolor=webcolors.rgb_to_hex(rgb), 
                             edgecolor=webcolors.rgb_to_hex(rgb), 
                             label=category) for category, rgb in category_and_rgb]
    return legend_elements

def open_image(idx,cfg):
    file_path = cfg.data.test.img_dir #'/opt/ml/nunbody/nunbody_segmentation/data/val/masks'
    file_names = os.listdir(file_path)
    file_names.sort()
    file_names = file_names
    src = os.path.join(file_path, file_names[idx])
    
    img = cv2.imread(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img)
    return img
def image_float_to_uint8(float_image):
    int_image = cv2.normalize(float_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return int_image

def draw_image(output,save_path,cfg):
    fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(16, 30), constrained_layout=True)
    length_output = len(output)
    legend_elements = make_legend_elements(category_names, class_color)
    a=0.4
    b=1-a    
    for row_num in range(5):
        origin_image = open_image(row_num,cfg)
        plt.imshow(origin_image)
        predict_mask = label_to_color_image(output[row_num])
        # mask_image = label_to_color_image(temp_masks[row_num])
        merge_image = cv2.addWeighted(image_float_to_uint8(origin_image),a,predict_mask.astype(np.uint8),b,0)
        
        ax[row_num][0].imshow(origin_image)
        ax[row_num][0].set_title(f"Orignal Image : {row_num}")
        ax[row_num][1].imshow(predict_mask)
        ax[row_num][1].set_title(f"Pred Mask : {row_num}")
        ax[row_num][2].imshow(merge_image)
        ax[row_num][2].set_title(f"Merge Mask : {row_num}")   
        ax[row_num][2].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0) 
    
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f'saved_{cfg.model.backbone.type}.jpg')
    print("saved image")
    
def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    checkpoint =  args.checkpoint  #'../work_dirs/_02.swin-t/coco swin/epoch_22.pth'
    cfg.gpu_ids = range(0, 1)
    cfg.data.test.test_mode = True

    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        drop_last=False,
        dist=False,
        shuffle=False)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    logit_outputs = single_gpu_test(model, data_loader)
    print("path:",args.save_path)
    draw_image(logit_outputs, args.save_path, cfg)

if __name__ == "__main__":
    print("start inference")
    main()