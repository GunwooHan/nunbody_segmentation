import os
import argparse

import torch
from dataset import CustomDataLoader
import cv2

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

def main(args):
    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_transform = A.Compose([
                            A.Resize(512, 512),
                            ToTensorV2()
                            ])

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # test dataset
    test_dataset = CustomDataLoader(args.data_path, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=1,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' +  directory)
    
    createFolder(args.save_path)

    def test(model, data_loader, device):
        model.eval()

        with torch.no_grad():
            for step, (imgs, image_infos, img_size) in enumerate(tqdm(test_loader)):
                imgs = torch.stack(imgs).to(device)

                model = model.to(device)
                outs = model(imgs)

                oms = torch.argmax(outs.squeeze(), dim=0).detach().cpu().numpy()
                oms[oms == 14] = 0
                oms = cv2.resize(oms, img_size[0], interpolation = cv2.INTER_NEAREST)
                
                cv2.imwrite(os.path.join(args.save_path , f'{image_infos[0]}.png'), oms)

    model_path = './d3_best_model.pt'

    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()

    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=15
    )

    model.load_state_dict(state_dict)
    model = model.to(device)

    # test set에 대한 prediction
    test(model, test_loader, device)

if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument('--data_path', type=str, nargs='?', default='/opt/ml/inbody/train_dataset/test')
     parser.add_argument('--save_path', type=str, nargs='?', default='./result')
     args = parser.parse_args()

     main(args)
