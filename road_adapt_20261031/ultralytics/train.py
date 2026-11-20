import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import torch

if __name__ == '__main__':
    model = RTDETR('SDI.yaml')
    torch.cuda.empty_cache()

    # model.load('') # loading pretrain weights
    model.train(data=r'myData.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4,
                workers=0,
                device=0,
                #resume='/media/sda1/renhonge/detr/RT-DETR/runs/train/my300/weights/last.pt', # last.pt path
                project='runs/train',
                name='RDD18SDI',
                # amp=True
                )