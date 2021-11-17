import os
import argparse

import cv2
import numpy as np
import torch
import pytorch_lightning as pl
import albumentations as A
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

if __name__ == '__main__':
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2471, 0.2435, 0.2616],
        ),
        ToTensorV2()
    ])
    a=0.4
    b=1-a
    
    dataset = PoseDataset(args.datadir, 'train', transform=transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, num_workers=4)
    print(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    for idx, img in enumerate(loader):
        origin_image = img[0][0].permute([1,2,0]).numpy()
        predict_mask = label_to_color_image(img[1][0].numpy())
        merge_image = cv2.addWeighted(image_float_to_uint8(origin_image),a,predict_mask.astype(np.uint8),b,0)
        
        
        cv2.imwrite(f'val_check/{idx:04d}.png', merge_image)