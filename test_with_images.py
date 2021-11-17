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

parser.add_argument('--seed', type=int, default=42)
# parser.add_argument('--train_datadir', type=str, default='data/train')
# parser.add_argument('--data_path', type=str, default='data/val/images')
parser.add_argument('--data_path', type=str, default='data/val/images')
parser.add_argument('--save_path', type=str, default='result')
parser.add_argument('--archi', type=str, default='DeepLabV3Plus')
parser.add_argument('--backbone', type=str, default='efficientnet-b4')
parser.add_argument('--pretrained_weights', type=str, default=None)
parser.add_argument('--color_output', type=bool, default=True)

args = parser.parse_args()


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


if __name__ == '__main__':

    model = SmpModel.load_from_checkpoint("saved/model.ckpt",
                                        args=args,
                                        train_transform=None,
                                        val_transform=None)

    test_transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2471, 0.2435, 0.2616],
        ),
        ToTensorV2()
    ])


    test_dataset = PoseDataset(args.data_path, 'test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.cuda()
    model.eval()

    for idx, (img, filename) in enumerate(test_loader):
        inputs = img.cuda()
        # tta_output = 

        output = model(inputs)
        iou_value = output.argmax(dim=1)
        target_mask = iou_value[0].detach().cpu().numpy()
        
        orig_img = cv2.imread(os.path.join(args.data_path, filename[0]))
        h, w, c = orig_img.shape
        target_mask = target_mask.astype(np.uint8)
        target_mask = cv2.resize(target_mask, (w,h), interpolation=cv2.INTER_NEAREST)

        filename = filename[0].split('.')[0]
        if args.color_output:
            color_output = label_to_color_image(target_mask)
            blended = cv2.addWeighted(orig_img, 0.5, color_output, (0.5), 0) 
            cv2.imwrite(f'{args.save_path}/{filename}.png', blended)
        else:
            cv2.imwrite(f'{args.save_path}/{filename}.png', target_mask)
        # break