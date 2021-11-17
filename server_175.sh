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
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b3
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b4
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b5
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b6
#python train.py --gpus 2 --ShiftScaleRotate 0.01 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b7
#python train.py --gpus 2 --ShiftScaleRotate 0.1 --ImageCompression 0.1 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b6
#python train.py --gpus 2 --ShiftScaleRotate 0.1 --ImageCompression 0.1 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone efficientnet-b5
#python train.py --gpus 2 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-tf_efficientnet_lite4 --train_datadir data/train_all
#python train.py --gpus 2 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-tf_efficientnet_lite4 --train_datadir data/train
#python train.py --gpus 2 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-tf_efficientnet_lite4 --fp16 32 --train_datadir data/train_all
#python train.py --gpus 2 --ShiftScaleRotate 0.2 --ImageCompression 0.01 --HorizontalFlip 0.5 --VerticalFlip 0.5 --backbone timm-tf_efficientnet_lite4 --fp16 32 --train_datadir data/train

#python train.py --gpus 2 --archi Unet --backbone timm-tf_efficientnet_lite4 --fp16 16 --train_datadir data/train
#python train.py --gpus 2 --archi UnetPlusPlus --backbone timm-tf_efficientnet_lite4 --fp16 16 --train_datadir data/train
#python train.py --gpus 2 --archi MAnet --backbone timm-tf_efficientnet_lite4 --fp16 16 --train_datadir data/train
#python train.py --gpus 2 --archi Linknet --backbone timm-tf_efficientnet_lite4 --fp16 16 --train_datadir data/train
#python train.py --gpus 2 --archi FPN --backbone timm-tf_efficientnet_lite4 --fp16 16 --train_datadir data/train
#python train.py --gpus 2 --archi PSPNet --backbone timm-tf_efficientnet_lite4 --fp16 16 --train_datadir data/train
#python train.py --gpus 2 --archi PAN --backbone timm-tf_efficientnet_lite4 --fp16 16 --train_datadir data/train
#python train.py --gpus 2 --archi DeepLabV3 --backbone timm-tf_efficientnet_lite4 --fp16 16 --train_datadir data/train
#python train.py --gpus 2 --archi DeepLabV3Plus --backbone timm-regnetx_064 --fp16 16 --train_datadir data/train

python train.py --gpus 2 --archi DeepLabV3Plus --backbone se_resnext101_32x4d --fp16 16 --train_datadir data/train_all --ShiftScaleRotate 0.9 --ImageCompression 0.1 --GridDistortion 0.1
