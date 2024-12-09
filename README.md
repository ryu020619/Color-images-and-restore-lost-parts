![데이콘](https://github.com/user-attachments/assets/97840d3d-a163-40e3-84d8-d193e1e602dc)
# Color-images-and-restore-lost-parts 🎨 
- - -

### 프로젝트 개요
+ ### *이미지 복원 ?*
이미지 복원 기술은 손상되거나 결손된 부분을 복구하고, 흑백 이미지에 자연스러운 색을 입히는 기술로,
역사적 사진 복원, 영상 편집, 의료 이미지 복구 등 다양한 분야에서 중요하게 활용되고 있다.

+ ### *DACON 데이터 설명*
train_input [폴더] : 흑백, 일부 손상된 PNG 학습 이미지 (input, 29603장)

train_gt [폴더] : 원본 PNG 이미지 (target, 29603장)

train.csv [파일] : 학습을 위한 Pair한 PNG 이미지들의 경로

test_input [폴더] : 흑백, 일부 손상된 PNG 평가 이미지 (input, 100장)

test.csv [파일] : 추론을 위한 Input PNG 이미지들의 경로

+ ### *복원 과정*
![슬라이드3](https://github.com/user-attachments/assets/8129af3f-5342-43f0-82c9-1187ee91c9e3)
---
![슬라이드4](https://github.com/user-attachments/assets/1c827562-ccee-425d-8c97-cbe861ede580)
---
![슬라이드5](https://github.com/user-attachments/assets/ec05d05a-ad1e-4c21-bf2f-6b876728db96)
---
![슬라이드6](https://github.com/user-attachments/assets/017ec2bb-c361-4135-8d6c-9d97b804796a)
---
![슬라이드7](https://github.com/user-attachments/assets/489d60b6-639d-4905-b92f-fb344a9656fa)
---
![슬라이드8](https://github.com/user-attachments/assets/ec58fe88-cf00-477a-baab-d0134bb6e6b0)
---

- - -
### test input
![test](https://github.com/user-attachments/assets/716e0842-b1eb-4241-8bca-dca31a1d6997)

- - -
###  epoch 30까지의 과정
- - - 
![epoch1](https://github.com/user-attachments/assets/d788f349-f4d0-49a2-b155-79336ba82d22)
### epoch 1
![epoch20](https://github.com/user-attachments/assets/9704b895-2e20-4988-a8ae-7fcc95b0d90f)
### epoch 20
![epoch29](https://github.com/user-attachments/assets/8b815998-68b5-474d-bdc2-98405fbdd962)
### epoch 30
---
-> 위 사진들과 같이 조금씩 손실 부분 영역과 색상이 복구 된 모습이다.

-> 대회 사이트는 다음 링크 참고
 ![이미지 색상화 및 손실 부분 복원 AI 경진대회]([https://github.com/gordicaleksa/pytorch-neural-style-transfer](https://dacon.io/competitions/official/236420/data))
 
- - -
### 사용한 코드
```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import zipfile
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import zipfile
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(1024 + 512, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(512 + 256, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.conv_block(256 + 128, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = self.conv_block(128 + 64, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        e5 = self.enc5(nn.MaxPool2d(2)(e4))

        d1 = self.dec1(torch.cat([self.up1(e5), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return torch.sigmoid(self.final(d4))

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class ImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None, limit=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
        self.transform = transform

        # 이미지 개수가 다르면 오류 발생
        assert len(self.input_images) == len(self.gt_images), "The number of images in gray and color folders must match"
        # 데이터 개수를 제한
        if limit:
            self.gt_images = self.gt_images[:limit]
            self.input_images = self.input_images[:limit]

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])
        input_image = cv2.imread(input_path)
        gt_image = cv2.imread(gt_path)
        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)
        return (
            torch.tensor(input_image).permute(2, 0, 1).float() / 255.0,
            torch.tensor(gt_image).permute(2, 0, 1).float() / 255.0
        )

generator = UNet().to(device)
discriminator = PatchGANDiscriminator().to(device)

# 모델을 GPU 0에 할당
generator = UNet().to(device)
discriminator = PatchGANDiscriminator().to(device)

adversarial_loss = nn.BCELoss()  
pixel_loss = nn.MSELoss()  

optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

train_dataset = ImageDataset("/home/work/RYU/open1.zip/open/train_input", "/home/work/RYU/open1.zip/open/train_gt", limit=None)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

epochs = 30
result_dir = "저장할 디렉토리 이름"
os.makedirs(result_dir, exist_ok=True)
checkpoint_path = "checkpoint.pth"

for epoch in range(epochs):
    generator.train()
    discriminator.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for input_images, gt_images in train_loader:
            input_images, gt_images = input_images.to(device), gt_images.to(device)

            real_labels = torch.ones_like(discriminator(gt_images)).to(device)
            fake_labels = torch.zeros_like(discriminator(input_images)).to(device)

            optimizer_G.zero_grad()
            fake_images = generator(input_images)
            pred_fake = discriminator(fake_images)

            g_loss_adv = adversarial_loss(pred_fake, real_labels)
            g_loss_pixel = pixel_loss(fake_images, gt_images)
            g_loss = g_loss_adv + 100 * g_loss_pixel
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            pred_real = discriminator(gt_images)
            loss_real = adversarial_loss(pred_real, real_labels)

            pred_fake = discriminator(fake_images.detach())
            loss_fake = adversarial_loss(pred_fake, fake_labels)

            d_loss = (loss_real + loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            running_loss_G += g_loss.item()
            running_loss_D += d_loss.item()

            pbar.set_postfix(generator_loss=g_loss.item(), discriminator_loss=d_loss.item())
            pbar.update(1)

    print(f"Epoch [{epoch+1}/{epochs}] - Generator Loss: {running_loss_G / len(train_loader):.4f}, Discriminator Loss: {running_loss_D / len(train_loader):.4f}")

    test_input_dir = "/home/work/RYU/open1.zip/open/test_input"
    output_dir = f"output_epoch_{epoch+1}"
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for img_name in sorted(os.listdir(test_input_dir)):
            img_path = os.path.join(test_input_dir, img_name)
            img = cv2.imread(img_path)
            input_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            output = generator(input_tensor).squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
            output = output.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, img_name), output)

    zip_filename = os.path.join(result_dir, f"epoch_{epoch+1}.zip")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for img_name in os.listdir(output_dir):
            zipf.write(os.path.join(output_dir, img_name), arcname=img_name)
    print(f"Epoch {epoch+1} results saved to {zip_filename}")

    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, checkpoint_path)

generator.train()  
discriminator.train()
```
