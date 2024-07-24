from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class CelebDataset(Dataset): 
    
    def __init__ (self, box_path, attr_path, landmarks_path,part_path,root_dir) -> None:
        
        self.to_tensor = transforms.ToTensor()
        self.data_info = pd.read_csv(box_path)
        self.root_dir = root_dir
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.attr_info = pd.read_csv(attr_path)
        self.land_info = pd.read_csv(landmarks_path)
        self.part_info = pd.read_csv(part_path)
        ### need to store info like bounding boxes for different stuff
        self.data_len = self.data_info.shape[0]
        
    def __getitem__(self, index) -> tuple :
        img_name = self.image_arr[index]
        img_as_img = Image.open((self.root_dir + img_name))
        #img_as_tensor = self.to_tensor(img_as_img)
        x_1: int = self.data_info.iloc[index]['x_1']
        y_1: int = self.data_info.iloc[index]['y_1']
        box = [x_1, y_1, x_1 + self.data_info.iloc[index]['width'], y_1 + self.data_info.iloc[index]['height']]
        
        return (box)

    def __len__(self) -> int:
        return self.data_len