import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import torch

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet18.yaml')
    torch.cuda.empty_cache()

    # model.load('') # loading pretrain weights
    model.train(data=r'SVRDD.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                workers=4,
                device='0',
                resume='runs/rtdetr-resnet18/weights/last.pt', # last.pt path
                project='/autodl-fs/data/RT-DETR/runs',
                name='rtdetr-resnet18',
                # amp=Tre
                optimizer='AdamW',
                lr0=0.0001,
                close_mosaic=10
                )