#python train.py --gpus 2 --backbone timm-resnest101e
#python train.py --gpus 2 --backbone timm-res2net50_26w_8s
#python train.py --gpus 2 --backbone timm-regnetx_120
#python train.py --gpus 2 --backbone se_resnet101
#python train.py --gpus 2 --backbone inceptionv4
#python train.py --gpus 2 --backbone timm-efficientnet-b6
#python train.py --gpus 2 --backbone efficientnet-b6
#python train.py --gpus 2 --backbone dpn92
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-mobilenetv3_small_minimal_100
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-mobilenetv3_small_100
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-mobilenetv3_small_075
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-mobilenetv3_large_minimal_100
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-mobilenetv3_large_100
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-mobilenetv3_large_075
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-efficientnet-lite4
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-efficientnet-lite3
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-efficientnet-lite2
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-efficientnet-lite1
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-efficientnet-lite0
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b0
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b1
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b2
# python train.py --gpus 2 --epochs 10 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b3
# python train.py --gpus 2 --epochs 15 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --RandomBrightnessContrast 0.1 --archi UnetPlusPlus --backbone timm-tf_efficientnet_lite4 # ver1
# python train.py --gpus 2 --epochs 15 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --RandomBrightnessContrast 0.1 --ElasticTransform 0.1 --RandomScale 1 --archi UnetPlusPlus --backbone timm-tf_efficientnet_lite4 # ver2
# python train.py --seed 42 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --RandomBrightnessContrast 0.1 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4

# python train.py --seed 42 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --RandomBrightnessContrast 0.1 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4
# python train.py --seed 43 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --RandomBrightnessContrast 0.1 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4 # ver3
# python train.py --seed 44 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --RandomBrightnessContrast 0.1 --learning_rate 0.0001 ---archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4
# python train.py --seed 42 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --RandomGamma 0.1 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4
# python train.py --seed 43 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --RandomGamma 0.1 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4
# python train.py --seed 44 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --RandomGamma 0.1 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4
# python train.py --seed 42 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4
# python train.py --seed 43 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4
# python train.py --seed 44 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4


# python train.py --gpus 2 --epochs 100 --ShiftScaleRotate 0.2 --ImageCompression 0.1 --RandomBrightnessContrast 0.1 --RandomScale 1 --archi UnetPlusPlus --backbone timm-tf_efficientnet_lite4 # ver2
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b4
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b5
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b6
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b7
# python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b6
# python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b5

# python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone xception
# python train.py --gpus 2 --epochs 8 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --RandomBrightnessContrast 0.05 --ElasticTransform 0.05 --img_size 1024 --backbone timm-efficientnet-lite4
# python train.py --gpus 2 --epochs 8 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --RandomBrightnessContrast 0.05 --ElasticTransform 0.05 --backbone se_resnext50_32x4d




# python train.py --seed 42 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --CLAHE 1.0 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4
# python train.py --seed 43 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --CLAHE 1.0 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4
# python train.py --seed 44 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --CLAHE 1.0 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4


# python train.py --seed 42 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --RandomResizedCrop 0.2 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4
# python train.py --seed 43 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --RandomResizedCrop 0.2 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4
# python train.py --seed 44 --gpus 2 --epochs 20 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --RandomResizedCrop 0.2 --learning_rate 0.0001 --archi DeepLabV3Plus --backbone timm-tf_efficientnet_lite4


python train.py --seed 42 --gpus 2 --epochs 50 --archi DeepLabV3Plus --backbone efficientnet-b4