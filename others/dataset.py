import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CustomDataLoader(Dataset):
        """COCO format"""
        def __init__(self, data_dir, mode = 'train', transform = None):
            super().__init__()
            self.mode = mode
            self.transform = transform
            self.data_dir = data_dir
            if mode in ['train', 'val']:
                train_dir = f'{data_dir}/train2014'
                self.train_file_list = os.listdir(train_dir)

                train_ann_dir = f'{data_dir}/train_mask'
                self.train_ann_file_list = os.listdir(train_ann_dir)

                val_dir = f'{data_dir}/val2014'
                self.val_file_list = os.listdir(val_dir)

                val_ann_dir = f'{data_dir}/val_mask'
                self.val_ann_file_list = os.listdir(val_ann_dir)
            elif mode == 'test':
                test_dir = f'{data_dir}'
                self.test_file_list = list(filter(lambda x : '.jpg' in x, os.listdir(test_dir)))
            
        def __getitem__(self, index: int):
            if self.mode == 'train':
                # dataset이 index되어 list처럼 동작
                image_file = os.path.splitext(self.train_file_list[index])[0]
                
                # cv2 를 활용하여 image 불러오기
                images = cv2.imread(f'{self.data_dir}/train2014/{image_file}.jpg')
                images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
                images /= 255.0

                masks = np.array(Image.open(f'{self.data_dir}/train_mask/{image_file}.png'))
                            
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    transformed = self.transform(image=images, mask=masks)
                    images = transformed["image"]
                    masks = transformed["mask"]
                return images, masks, image_file

            if self.mode == 'val':
                # dataset이 index되어 list처럼 동작
                image_file = os.path.splitext(self.val_file_list[index])[0]
                
                # cv2 를 활용하여 image 불러오기
                images = cv2.imread(f'{self.data_dir}/val2014/{image_file}.jpg')
                images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
                images /= 255.0

                masks = np.array(Image.open(f'{self.data_dir}/val_mask/{image_file}.png'))
                            
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    transformed = self.transform(image=images, mask=masks)
                    images = transformed["image"]
                    masks = transformed["mask"]
                return images, masks, image_file
            
            if self.mode == 'test':
                image_file = os.path.splitext(self.test_file_list[index])[0]
                images = cv2.imread(os.path.join(self.data_dir, f'{image_file}.jpg'))
                images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
                images /= 255.0
                w, h, _ = images.shape
                # transform -> albumentations 라이브러리 활용
                if self.transform is not None:
                    transformed = self.transform(image=images)
                    images = transformed["image"]
                return images, image_file, (h, w)
        
        def __len__(self) -> int:
            # 전체 dataset의 size를 return
            if self.mode == 'train':
                return len(self.train_file_list)
            if self.mode == 'val':
                return len(self.val_file_list)
            if self.mode == 'test':
                return len(self.test_file_list)
