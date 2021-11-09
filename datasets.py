import os

import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        if mode == 'train' or 'val':
            self.images = os.listdir(os.path.join(data_dir, 'images'))
            self.masks = os.listdir(os.path.join(data_dir, 'masks'))
        elif mode == 'test':
            self.images = os.listdir(os.path.join(data_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if self.mode == 'train' or 'val':
            img = cv2.imread(os.path.join(self.data_dir, 'images', self.images[item]))
            mask = cv2.imread(os.path.join(self.data_dir, 'masks', self.masks[item]), 0)

            if self.transform is not None:
                transformed = self.transform(image=img, mask=mask)
                img = transformed["image"]
                mask = transformed["mask"]
                return img, mask.long()

            return img, mask

        elif self.mode == 'test':
            img = cv2.imread(os.path.join(self.data_dir, self.images[item]))
            
            if self.transform is not None:
                transformed = self.transform(image=img)
                img = transformed["image"]
                return img

            return img

        # h,w,c = img.shape
        # h_flag = h%2==1
        # w_flag = w%2==1
        # if h_flag:
        #     temp_transformed = A.Resize(w, h+1)(image=img, mask=mask)
        #     img = temp_transformed["image"]
        #     mask = temp_transformed["mask"]
        # elif w_flag:
        #     temp_transformed = A.Resize(w+1, h)(image=img, mask=mask)
        #     img = temp_transformed["image"]
        #     mask = temp_transformed["mask"]



if __name__ == '__main__':
    transforms = A.Compose([
        A.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2471, 0.2435, 0.2616],
        ),
        ToTensorV2()
    ])
    ds = PoseDataset('data/train', mode='train', transform=transforms)

    print()