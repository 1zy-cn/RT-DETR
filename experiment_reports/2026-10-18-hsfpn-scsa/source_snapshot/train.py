import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import torch

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/resnet18-HSFPN+SCSA.yaml')
    torch.cuda.empty_cache()

    # model.load('') # loading pretrain weights
    model.train(data=r'SVRDD.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                workers=4,
                device='0',
                #resume='runs/rtdetr-resnet18-HSFPN/weights/last.pt', # last.pt path
                project='/autodl-fs/data/RT-DETR/runs',
                name='resnet18-HSFPN+SCSAEnable',
                # amp=Tre
                optimizer='AdamW',
                lr0=0.0001,
                close_mosaic=10
                )