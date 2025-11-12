import warnings

import torch

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"F:\yolo+ir+rgb\YOLOv11_RGBT_master0619\YOLOv11_RGBT0\ultralytics\my_cfg\models\ECFFN-ablation\yolo11n-MMFD.yaml")
    model.load('yolo11n.pt') # loading pretrain weights
    model.train(data=R'F:/yolo+ir+rgb/YOLOv11_RGBT_master0619/YOLOv11_RGBT/ultralytics/my_cfg/data/M3FD_RGBT.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # use_simotm="RGB",
                # channels=3,
                use_simotm="RGBRGB6C",
                channels=6,
                project='runs/M3FD',
                name='M3FD-n-yolo11n-MMFD',
                )
    del model
    torch.cuda.empty_cache()

