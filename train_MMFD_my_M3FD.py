import warnings

import torch

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"F:\yolo+ir+rgb\YOLOv11_RGBT_master0619\YOLOv11_RGBT0\ultralytics\cfg\models\11-RGBT\yolo11-RGBT-latefusion.yaml")
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
                name='M3FD-n-yolo11-RGBT-latefusion',
                )
    del model
    torch.cuda.empty_cache()



    # Step 2
    # model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion-MCF.yaml')
    # model.train(data=R'ultralytics/cfg/datasets/M3FD.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=1,
    #             batch=16,
    #             close_mosaic=0,
    #             workers=2,
    #             device='0',
    #             optimizer='SGD',  # using SGD
    #             # resume='', # last.pt path
    #             # amp=False, # close amp
    #             fraction=0.01,  # 仅用 1% 的数据训练, 快速得到一个模型权重模板    Train with only 1% of the data. Quickly obtain a model weight template
    #             use_simotm="RGBRGB6C",
    #             channels=6,
    #             project='runs/M3FD',
    #             name='M3FD-yolo11n-RGBT-midfusion-MCF-e300-16-',
    #             )
    # del model
    # torch.cuda.empty_cache()

    # Step 3     python transform_MCF.py


    # Step 4

    # model = YOLO(r'M3FD-yolo11n-RGBT-midfusion-MCF.pt')
    # model.train(data=R'ultralytics/cfg/datasets/M3FD.yaml',
    #             cache=False,
    #             imgsz=640,
    #             epochs=100,
    #             batch=16,
    #             close_mosaic=0,
    #             workers=2,
    #             device='0',
    #             optimizer='SGD',  # using SGD  微调参数请参考论文进行设置，事实上，论文中的参数大概率也不是最佳参数，我们对超参数的选取没有做大量测试。仅做了几组可行的参数设置
    #             # For fine-tuning the parameters, please refer to the paper for setting. In fact, the parameters in the paper are probably not the optimal ones. We did not conduct extensive tests on the selection of hyperparameters. We only made a few sets of feasible parameter settings.
    #
    #             # resume='', # last.pt path
    #             # amp=False, # close amp
    #             # fraction=0.2,
    #             freeze=[2, 3, 4, 5, 6, 17, 18, 23, 24, 29, 30],
    #             use_simotm="RGBRGB6C",
    #             channels=6,
    #             project='runs/M3FD',
    #             name='M3FD-yolo11n-RGBT-midfusion-MCF-final-e300-16-',
    #             )
    # del model
    # torch.cuda.empty_cache()

