import os

import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        if mode == 'train' or 'val':
            self.images = sorted(os.listdir(os.path.join(data_dir, 'images')))
            self.masks = sorted(os.listdir(os.path.join(data_dir, 'masks')))
            # if self.images[0].startswith('.'): self.images=self.images[1:] #주피터 파일 제거
            # if self.masks[0].startswith('.'): self.masks=self.masks[1:] #주피터 파일 제거
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
                return img, mask.long(), self.images[item]

            return img, mask, self.images[item]

        elif self.mode == 'test':
            img = cv2.imread(os.path.join(self.data_dir, self.images[item]))
            
            if self.transform is not None:
                transformed = self.transform(image=img)
                img = transformed["image"]
                return img, self.images[item]

            return img, self.images[item]

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
    ds = TestDataset('data/train', mode='train', transform=transforms)

    print()