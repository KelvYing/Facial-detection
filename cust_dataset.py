from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from mask import create_mask


class FaceDataset(Dataset): 
    
    def __init__ (self, box_path, root_dir, transform = None) -> None:
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.data_info = box_path
        self.root_dir = root_dir
        self.image_arr = np.asarray(self.data_info.groupby('image_name').apply(lambda x: x.to_dict(orient='list')).tolist())
        
        ### need to store info like bounding boxes for different stuff
        self.data_len = self.image_arr.shape[0]
        
    def __getitem__(self, index) -> tuple :
        
        img_dat = self.image_arr[index]
        img_name = img_dat['image_name'][0]
        
        img = Image.open((self.root_dir + img_name))
        if self.transform:
            img = self.transform(img)
        img_as_img = np.array(img)
        print(type(img_as_img))
        img_as_tensor = self.to_tensor(img_as_img)
        
        #store bounding box information
        box = torch.tensor(list(zip(img_dat['x0'], img_dat['y0'], img_dat['x1'], img_dat['y1']))[0], dtype=torch.float32)
        print(type(box))
        
        return img_as_tensor , box , create_mask(box, img_as_tensor)

    def __len__(self) -> int:
        return self.data_len