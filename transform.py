import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import DualTransform


class PoseHorizontalFlip(DualTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.transform = A.HorizontalFlip(always_apply=True)

    def apply(self, img, **params):
        # print(img.shape)
        if len(img.shape) == 2:
            mask = self.transform(image=img)['image']

            temp_mask = np.zeros_like(mask)

            temp_mask[mask == 1] = 1
            temp_mask[mask == 14] = 14

            # hand
            temp_mask[mask == 2] = 3
            temp_mask[mask == 3] = 2

            # foot
            temp_mask[mask == 4] = 5
            temp_mask[mask == 5] = 4

            # thigh
            temp_mask[mask == 6] = 7
            temp_mask[mask == 7] = 6

            # calf
            temp_mask[mask == 6] = 7
            temp_mask[mask == 7] = 6

            # thigh
            temp_mask[mask == 8] = 9
            temp_mask[mask == 9] = 8

            # thigh
            temp_mask[mask == 10] = 11
            temp_mask[mask == 11] = 10

            # thigh
            temp_mask[mask == 12] = 13
            temp_mask[mask == 13] = 12
            return temp_mask
        else:
            img = self.transform(image=img)['image']
            return img


def make_transform(args):
    base_transform = [
        # A.CLAHE(clip_limit=(1, 4),
        #                     tile_grid_size=(8, 8),
        #                     p=1.0
        #                     ),
        A.Resize(args.img_size, args.img_size),
        A.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2471, 0.2435, 0.2616],
        ),
        ToTensorV2()
    ]

    train_transform = []
    #
    # if args.RandomScale:
    #     train_transform.append(A.RandomScale([0.5, 2], p=1))

        # RandomBrightnessContrast, HueSaturationValue, RGBShift, RandomGamma 모두 색상/밝기/감마/대비 변경



    if args.Blur:
        train_transform.append(
            A.Blur(p=args.Blur))
    if args.Blur:
        train_transform.append(
            A.ElasticTransform(p=args.Blur))

    if args.CLAHE:
        train_transform.append(A.CLAHE(clip_limit=(1, 4),
                                       tile_grid_size=(8, 8),
                                       p=args.CLAHE
                                       ))
    if args.RandomBrightnessContrast:
        train_transform.append(
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       brightness_by_max=True,
                                       p=args.RandomBrightnessContrast
                                       ))
    if args.HueSaturationValue:
        train_transform.append(A.HueSaturationValue(hue_shift_limit=20,
                                                    sat_shift_limit=30,
                                                    val_shift_limit=20,
                                                    p=args.HueSaturationValue
                                                    ))
    if args.RGBShift:
        train_transform.append(A.RGBShift(r_shift_limit=20,
                                          g_shift_limit=20,
                                          b_shift_limit=20,
                                          p=args.RGBShift
                                          ))
    if args.RandomGamma:
        train_transform.append(A.RandomGamma(gamma_limit=(80, 120),
                                             p=args.RandomGamma
                                             ))
    if args.HorizontalFlip:
        train_transform.append(PoseHorizontalFlip(p=args.HorizontalFlip))

    if args.VerticalFlip:
        train_transform.append(A.VerticalFlip(p=args.VerticalFlip))

    if args.ShiftScaleRotate:
        train_transform.append(A.ShiftScaleRotate(shift_limit=0.2,
                                                  scale_limit=0.2,
                                                  rotate_limit=10,
                                                  border_mode=args.ShiftScaleRotateMode,
                                                  p=args.ShiftScaleRotate
                                                  ))
    if args.GridDistortion:
        train_transform.append(A.GridDistortion(num_steps=5,
                                                distort_limit=(-0.3, 0.3),
                                                p=args.GridDistortion
                                                ))
    if args.MotionBlur:
        train_transform.append(A.MotionBlur(blur_limit=(3, 7),
                                            p=args.MotionBlur
                                            ))
    if args.RandomResizedCrop:
        train_transform.append(A.RandomResizedCrop(height=args.img_size,
                                                   width=args.img_size,
                                                   scale=(-0.4, 1.0),
                                                   ratio=(0.75, 1.3333333333333333),
                                                   p=args.RandomResizedCrop
                                                   ))
    if args.ImageCompression:
        train_transform.append(A.ImageCompression(quality_lower=99,
                                                  quality_upper=100,
                                                  p=args.ImageCompression
                                                  ))
    train_transform.extend(base_transform)

    train_transform = A.Compose(train_transform)
    test_transform = A.Compose(base_transform)

    return train_transform, test_transform
