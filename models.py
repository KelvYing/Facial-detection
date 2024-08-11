from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms, DataLoader
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model
vgg_backbone = models.vgg16(pretrained = True)
vgg_backbone_classifier = nn.Sequential()
for param in vgg_backbone.parameters():
    param.requires_grad = False
vgg_backbone.eval().to(device) 


class RCNN(nn.Module):
    def __init__(self):
        super.__init__()
        feature_dim = 25088
        self.backbone = vgg_backbone
        self.bbox = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Tanh(),
        )
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()
        
    def forward(self, x):
        feat = self.backbone(x)
        bbox = self.bbox(x)
        return bbox

    def calc_loss(self, probs, _delta , labels, deltas):
        ...