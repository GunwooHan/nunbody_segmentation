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

class_color = [[0, 0, 0], [192, 0, 128],[0, 128, 192],[0, 128, 64],[128, 0, 0],\
        [64, 0, 128],[64, 0, 192],[192, 128, 64],[192, 192, 128],[64, 64, 128],\
        [128, 0, 192],[255, 0, 0],[0, 255, 0],[0, 0, 255],[128, 128, 128]]
category_names = ['background', 'body', 'R-hand', 'L-hand','L-foot','R-foot', 'R-thigh', 'L-thigh',  'R-calf','L-calf',  'L-arm', 'R-arm', 'L-forearm','R-forearm','head']

def parse_args():
    parser = argparse.ArgumentParser(description='Inference and make the csv file for submission to AI-Stage')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    # parser.add_argument('--save_path', default = 'inference_image', help='output file path to save results')
    
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

def draw_image(data_loader,cfg):
    fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(16, 30), constrained_layout=True)
    legend_elements = make_legend_elements(category_names, class_color)
    a=0.4
    b=1-a
    for idx, img in enumerate(data_loader):
        img_meta, image, gt_semantic_seg = img.values()
        origin_image = image.data[0][0].permute([1,2,0]).numpy()
        origin_mask=np.where(gt_semantic_seg.data[0][0,0]==255,0,gt_semantic_seg.data[0][0,0])
        merge_image = cv2.addWeighted(image_float_to_uint8(origin_image),a,label_to_color_image(origin_mask).astype(np.uint8),b,0)
        
        cv2.imwrite(f'val_check/{idx:04d}.png', label_to_color_image(origin_mask).astype(np.uint8))# merge_image)
        # for i in range(14):
        #     zz = np.where(np.stack((origin_mask,origin_mask,origin_mask),axis=2)==i+1,255,image_float_to_uint8(origin_image))
        #     cv2.imwrite(f'val_check/{idx:04d}{category_names[i+1]}_{i+1}.png', zz)
def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    checkpoint =  args.checkpoint  #'../work_dirs/_02.swin-t/coco swin/epoch_22.pth'
    cfg.gpu_ids = range(0, 1)
    cfg.data.test.test_mode = False

    # build dataset & dataloader
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        drop_last=False,
        dist=False,
        shuffle=False)

    draw_image(data_loader, cfg)

if __name__ == "__main__":
    print("start inference")
    main()
    print("finish")