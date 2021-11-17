import os
import random
import warnings
import argparse 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
# from utils import label_accuracy_score, add_hist
from dataset import CustomDataLoader

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

def main(args):
    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 8   # Mini-batch size
    num_epochs = 20
    learning_rate = 0.0001

    # seed 고정
    random_seed = 1995
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    category_names = ["Backgroud", "Body", "Right Hand", "Left Hand", "Left Foot", "Right Foot", "Right Thigh",
                    "Left Thigh", "Right Calf", "Left Calf", "Left Arm", "Right Arm", "Left Forearm", "Right Forearm", "Head"]

    
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_transform = A.Compose([
                                A.Resize(512, 512),
                                A.Compose([A.RandomScale(scale_limit=(-0.5, 0.5), p=0.3, interpolation=1),
                                A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT),
                                A.Resize(512, 512, cv2.INTER_NEAREST), ]),
                                ToTensorV2()
                                ])

    val_transform = A.Compose([
                                A.Resize(512, 512),
                                ToTensorV2()
                                ])

    # train dataset
    train_dataset = CustomDataLoader(args.data_path, mode='train', transform=train_transform)

    # validation dataset
    val_dataset = CustomDataLoader(args.data_path, mode='val', transform=val_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    # model 불러오기
    # 출력 label 수 정의 (classes=11)
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=15,                     # model output channels (number of classes in your dataset)
    )

    def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
        print(f'Start training..')
        n_class = 15
        best_loss = 9999999
        
        for epoch in range(num_epochs):
            model.train()

            hist = np.zeros((n_class, n_class))
            for step, (images, masks, _) in enumerate(data_loader):
                images = torch.stack(images)       
                masks = torch.stack(masks).long() 
                
                # gpu 연산을 위해 device 할당
                images, masks = images.to(device), masks.to(device)
                
                # device 할당
                model = model.to(device)
                
                # inference
                outputs = model(images)
                
                # loss 계산 (cross entropy loss)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                
                # step 주기에 따른 loss 출력
                if (step + 1) % 25 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                            Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                
            # validation 주기에 따른 loss 출력 및 best model 저장
            if (epoch + 1) % val_every == 0:
                avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
                if avrg_loss < best_loss:
                    print(f"Best performance at epoch: {epoch + 1}")
                    print(f"Save model in {saved_dir}")
                    best_loss = avrg_loss
                    save_model(model, saved_dir)

    def validation(epoch, model, data_loader, criterion, device):
        print(f'Start validation #{epoch}')
        model.eval()

        with torch.no_grad():
            n_class = 15
            total_loss = 0
            cnt = 0
            
            hist = np.zeros((n_class, n_class))
            for step, (images, masks, _) in enumerate(data_loader):
                
                images = torch.stack(images)       
                masks = torch.stack(masks).long()  

                images, masks = images.to(device), masks.to(device)            
                
                # device 할당
                model = model.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
            
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
            
            avrg_loss = total_loss / cnt
            print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                    mIoU: {round(mIoU, 4)}')
            print(f'IoU by class : {IoU_by_class}')
            
        return avrg_loss

    # 모델 저장 함수 정의
    val_every = 1

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    def save_model(model, saved_dir, file_name='d3_best_model.pt'):
        check_point = {'net': model.state_dict()}
        output_path = os.path.join(saved_dir, file_name)
        torch.save(model, output_path)

    # Loss function 정의
    criterion = nn.CrossEntropyLoss()

    # Optimizer 정의
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)

    train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, device)

if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument('--data_path', type=str, nargs='?', default='/opt/ml/inbody/train_dataset') # image 폴더의 상위 폴더를 입력
     args = parser.parse_args()

     main(args)