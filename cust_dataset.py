from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np



class FaceDataset(Dataset): 
    
    def __init__ (self, box_path, root_dir) -> None:
        
        self.to_tensor = transforms.ToTensor()
        self.data_info = box_path
        self.root_dir = root_dir
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        
        ### need to store info like bounding boxes for different stuff
        self.data_len = self.data_info.shape[0]
        
    def __getitem__(self, index) -> tuple :
        
        img_name = self.image_arr[index]
        
        
        img_as_img = np.array(Image.open((self.root_dir + img_name)))
        print(type(img_as_img))
        img_as_tensor = self.to_tensor(img_as_img)
        
        #store bounding box information
        x_0: int = self.data_info.iloc[index]['x0']
        y_0: int = self.data_info.iloc[index]['y0']
        x_1: int = self.data_info.iloc[index]['x1']
        y_1: int = self.data_info.iloc[index]['y1']
        box = (x_0, y_0, x_1, y_1)
        
        return (box , img_as_tensor)

    def __len__(self) -> int:
        return self.data_len