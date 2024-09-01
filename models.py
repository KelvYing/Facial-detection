import torch.nn as nn
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG

from cust_dataset import YOLODataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model
vgg_backbone = models.vgg16(weights = 'VGG16_Weights.DEFAULT')

#vgg_backbone = vgg_backbone.features
vgg_backbone = nn.Sequential(*list(vgg_backbone.features.children())[:-1])

for param in vgg_backbone.parameters():
    param.requires_grad = False
vgg_backbone.eval().to(device) 


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        feature_dim = 512 * 7 * 7
        self.backbone = vgg_backbone
        self.bbox = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 4, kernel_size=1),
            nn.Linear(4,4),
        )
        self.sl1 = nn.L1Loss()
        self.mse = nn.MSELoss() 
        
    def forward(self, x):
        feat = self.backbone(x)
        bbox = self.bbox(feat)
        return bbox

class CustomYOLO(YOLO):
    def get_dataloader(self, dataset_path, batch_size, workers , rank = 0, mode = 'train'):
        
        dat = YOLODataset()
        
        return DataLoader(dat, batch_size = batch_size , num_workers = workers, shuffle = True)