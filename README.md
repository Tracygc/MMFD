# MMFD
MMFD: Modality-aware and Multi-scale Feature Fusion Network for Multispectral Fusion Detection


### Dataset Structure
```
dataset/
├── images/
│   ├── visible/
│   │   ├── train/  # Store training visible images
│   │   └── val/    # Store validation visible images
│   └── infrared/
│       ├── train/  # Store training infrared images
│       └── val/    # Store validation infrared images
└── labels/
    ├── visible/
    │   ├── train/  # Store training visible image labels
    │   └── val/    # Store validation visible image labels
    └── infrared/
        ├── train/  # Store training infrared image labels
        └── val/    # Store validation infrared image labels
---------------------------------------------------------------------
# FLIR_aligned.yaml  (for aligned iimages or visible images)

train: G:/datasets/FLIR-align-3class/FLIR-align-3class/images/visible/train # 128 images
val: G:/datasets/FLIR-align-3class/FLIR-align-3class/images/visible/val # 128 images


# number of classes
nc: 3

# class names
names: ["person", "car", "bicycle"]
-----------------------------------------------------------------------
# FLIR_aligned_IF.yaml  (for infrared images)

train: G:/datasets/FLIR-align-3class/FLIR-align-3class/images/infrared/train # 128 images
val: G:/datasets/FLIR-align-3class/FLIR-align-3class/images/infrared/val # 128 images


# number of classes
nc: 3

# class names
names: ["person", "car", "bicycle"]
-----------------------------------------------------------------------

```
### Install Dependencies
Install the ultralytics package, including all requirements, in a Python>=3.8 environment with PyTorch>=1.8.
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install ultralytics
```
### Python
MCF train: 
Take the M3FD dataset as an example, and proceed with the following steps for training and testing in sequence.        
```
python train_ECFFN_my_M3FDstep1.py
python train_ECFFN_my_M3FDstep2.py.py
python train_ECFFN_my_M3FDstep3.py.py
python train_ECFFN_my_M3FDstep4.py.py
python val_M3FD.py
```

MMFD train: 
Take the M3FD dataset as an example, and proceed with the following steps for training and testing in sequence.        
```
python train_MMFD_my_M3FD.py
python val_M3FD.py
```

### Reference Links
```
https://docs.ultralytics.com/
https://github.com/wandahangFY/YOLOv11-RGBT
```
