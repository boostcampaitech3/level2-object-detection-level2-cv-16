# Team Medic(CV-16)

---

## Project Overview

- Project Period
2022.03.21 ~ 2022.04.08
- Project Wrap Up Report
[Object Det_CV-16_Medic_팀 리포트](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ff907b45-2d26-45df-9c24-7b0db6a1b00c/Object_Det_CV_팀_리포트(16조).pdf)
<br>

## ♻️ 재활용 품목 분류를 위한 Object Detection 🗑

### 😎 Members

| 권순호 | 서다빈 | 서예현 | 이상윤 | 전경민 |
| --- | --- | --- | --- | --- |
| [Github](https://github.com/tnsgh9603) | [Github](https://github.com/sodabeans) | [Github](https://github.com/yehyunsuh) | [Github](https://github.com/SSANGYOON?tab=repositories) | [Github](https://github.com/seoulsky-field) |
<br>

### 🌏 Contributions

- 권순호: Faster-RCNN, Cascade-RCNN, Swin-Large,Step, LRscheduler, Ensemble
- 서다빈: Cascade-RCNN, Swin-T ,CosineAnnealing, FocalLoss, YOLOV5, Optimizer 비교(SGD,Adam,AdamW), TTA, Ensemble
- 서예현: YOLOX, UniverseNet,  various IoU Losses, TTA, NMS & WBF Ensemble
- 이상윤:  stratified k-fold train/valid split 구현 , Swin-large + cascade r-cnn 구현, Efficientdet
- 전경민: Data Augmentation 및 Data Handling, 각종 Experiments 진행
<br>

### **❓Problem Definition**

- 바야흐로 **대량 생산, 대량 소비**의 시대. 우리는 많은 물건이 대량으로 생산되고 소비되는 시대를 삶에 따라 **쓰레기 대란, 매립지 부족**과 같은 사회 문제 발생

![image](https://user-images.githubusercontent.com/83350060/163540616-ef692c19-b993-4152-ab7e-132c6faf2df1.png)

- 버려지는 쓰레기 중 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문에 분리수거는 사회적 환경 부담 문제를 줄일 수 있는 방법
- Deep Learning을 통해 쓰레기들을 자동으로 분류할 수 있는 모델 개발
- 쓰레기를 줍는 드론, 쓰레기 배출 방지 비디오 감시, 인간의 쓰레기 분류를 돕는 AR 기술과 같은 여러 기술을 통해서 조금이나마 개선이 가능할 것으로 기대
<br>

### 💾 Datasets

- Number of Classes: 10
- Number of datasets: 9754
- Labels: General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- Image Size: (1024, 1024)
- train: 4883장의 train image 존재
- test: 4871장의 test image 존재
- train.json: train image에 대한 annotation file (coco format)
- test.json: test image에 대한 annotation file (coco format)
<br>

### 💾 Annotation Files

- COCO format으로 구성(images&annotations)
- images
    - id: 파일 안에서 image 고유 id, ex) 1
    - height: 1024
    - width: 1024
    - file*name: ex) train*/002.jpg
- annotations
    - id: 파일 안에 annotation 고유 id, ex) 1
    - bbox: 객체가 존재하는 박스의 좌표 (x*min, y*min, w, h)
    - area: 객체가 존재하는 박스의 크기
    - category_id: 객체가 해당하는 class의 id
    - image_id: annotation이 표시된 이미지 고유 id
<br>

### 💻 **Development Environment**

- GPU: Tesla V100
- OS: Ubuntu 18.04.5LTS
- CPU: Intel Xeon
- Python : 3.8.5
<br>

### 📁 Project Structure

```markdown
├── mmdetection
│   ├── faster_rcnn_train.ipynb
│   ├── faster_rcnn_inference.ipynb
|   ├── config
│   └── etc.
├── efficientdet_train.ipynn
├── efficientdet_inference.ipynb
├── yolov5_files
│   ├── manifest.txt
│   ├── trash.yaml
│   ├── voc.names
│   ├── yolov5.ipynb
│   └── yolov5_sumbmission.ipynb
├── ensemble.ipynb
├── fiftyone_train.py
├── inference.py
├── stratifiedkfold.py
├── submission_to_json.ipynb
├── valid_split.py
└── visualization.py
```
<br>

### 👨‍🏫 Evaluation Methods

<img src="https://latex.codecogs.com/png.image?\inline&space;\huge&space;\dpi{100}\bg{white}\textup{Precision}=\frac{TP}{TP&plus;FP}=\frac{TP}{\textup{All&space;Detections}}">

<img src="https://latex.codecogs.com/png.image?\inline&space;\huge&space;\dpi{100}\bg{white}\textup{Recall}=\frac{TP}{TP&plus;FN}=\frac{TP}{\textup{All&space;Ground&space;Truths}}">

<img src="https://latex.codecogs.com/png.image?\inline&space;\huge&space;\dpi{100}\bg{white}\textup{mAP}=\frac{1}{n}\sum_{k=1}^{n}AP_{k}&space;\textup{&space;where&space;n&space;means&space;each&space;class}">

### 💯 Final Score

![image](https://user-images.githubusercontent.com/83350060/163540703-cfe5fb39-0638-4fb4-8628-756f3137e216.png)

ensemble result of :
Faster R-CNN, Cascade R-CNN(Swin Transformer Tiny/Large), YOLOv5, YOLOX, UniverseNet
<br>

## 👀 How to Start

- downloading the github repository

```powershell
git clone https://github.com/boostcampaitech3/level2-object-detection-level2-cv-16.git
```

- installing mmdetection library

```powershell
conda install pytorch=1.7.1 cudatoolkit=11.0 torchvision -c pytorch

pip install openmim

mim install mmdet
```

- training the model

```powershell
cd level2-object-detection-level2-cv-16/mmdetection

python tools/train.py <<directory_of_the_config_file>>
```
<br>

### 📄 Experiments & Submission Report

[Notion](https://www.notion.so/W10-12-Object-Detection-Project-Team-Medic-027299e9ecb64d01aa5dcbc07307aef0)
