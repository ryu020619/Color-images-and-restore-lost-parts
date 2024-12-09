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
