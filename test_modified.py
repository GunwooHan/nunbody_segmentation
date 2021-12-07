import os
import argparse
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import SmpModel
from datasets_test import TestDataset
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import webcolors
from model import SmpModel

class_color = [[0, 0, 0], [192, 0, 128],[0, 128, 192],[0, 128, 64],[128, 0, 0],\
        [64, 0, 128],[64, 0, 192],[192, 128, 64],[192, 192, 128],[64, 64, 128],\
        [128, 0, 192],[255, 0, 0],[0, 255, 0],[0, 0, 255],[128, 128, 128]]
category_names = ['background', 'body','R-hand', 'L-hand','L-foot','R-foot', 'R-thigh', 'L-thigh',  'R-calf','L-calf',  'L-arm', 'R-arm', 'L-forearm','R-forearm','head']


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
# parser.add_argument('--train_datadir', type=str, default='data/train')
parser.add_argument('--test_datadir', type=str, default='data/val')
parser.add_argument('--archi', type=str, default='Unet')
parser.add_argument('--backbone', type=str, default='timm-mobilenetv3_small_100')
parser.add_argument('--pretrained_weights', type=str, default=None)
parser.add_argument('--load_weights', type=str, default=None)
parser.add_argument('--color_output', type=bool, default=True)

args = parser.parse_args()

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

def draw_image(origin_image, gt_mask, predict_mask, loss, mIoU, img_id):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 30), constrained_layout=True)
    legend_elements = make_legend_elements(category_names, class_color)
    a=0.4
    b=1-a
    predict_mask = label_to_color_image(predict_mask)
    merge_image = cv2.addWeighted(image_float_to_uint8(origin_image),a,predict_mask.astype(np.uint8),b,0)
    ax[1][0].imshow(origin_image)
    ax[1][0].set_title(f"Orignal Image : {img_id}")
    ax[1][1].imshow(predict_mask)
    ax[1][1].set_title(f"Pred Mask : {img_id}")
    ax[1][2].imshow(merge_image)
    ax[1][2].set_title(f"Merge Mask : {img_id}")   
    ax[1][2].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.savefig(f'result/{loss:0.4f}_{mIoU:0.2f}_{img_id}.jpg')
    print("saved image")            

if __name__ == '__main__':
    model_path = args.load_weights # 'saved/Unet_timm-tf_efficientnet_lite4-epoch=10-val/mIoU=0.05-v1.ckpt' 
    model = SmpModel.load_from_checkpoint(model_path,
                                        args=args,
                                        train_transform=None,
                                        val_transform=None)

    test_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2471, 0.2435, 0.2616],
        ),
        ToTensorV2()
    ])

    dataset = TestDataset(args.test_datadir, 'val', transform=test_transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.cuda()
    model.eval()
    
    for idx, img in enumerate(loader):
        image = img[0].cuda()
        gt_mask = img[1][0]
        output = model(image)
        iou_value = output.argmax(dim=1)
        predict_mask = iou_value[0].detach().cpu().numpy()
        draw_image(image.cpu().numpy(), gt_mask, predict_mask, loss = 0.999, mIoU = 10.12, img_id = img[2][0])
        if idx==100:
            break