from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from mask import create_mask

def normalize_boxes(boxes, image_width, image_height):
    print(boxes.shape)
    return (boxes / torch.tensor([image_width, image_height, image_width, image_height])).squeeze()

# Don't forget to denormalize when visualizing or evaluating:
def denormalize_boxes(normalized_boxes, image_width, image_height):
    return normalized_boxes * torch.tensor([image_width, image_height, image_width, image_height])


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
        img_as_tensor = self.to_tensor(img_as_img)
        
        #store bounding box information
        box = torch.tensor(list(zip( img_dat['x0'], img_dat['y0'], img_dat['x1'], img_dat['y1']))[0], dtype=torch.float32)
                
        scale = torch.tensor( [ 224 / img_dat['width'][0] , 224/ img_dat['height'][0] ,
                                224 / img_dat['width'][0] , 224/ img_dat['height'][0]])
        scaled_box = box * scale
        #normalized_box = normalize_boxes(box.unsqueeze(0), img_dat['width'][0], img_dat['height'][0])
        
        return img_as_tensor , scaled_box #, create_mask(box, img_as_tensor)

    def __len__(self) -> int:
        return self.data_len
    
class YOLODataset(Dataset): 
    
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
        img_as_tensor = self.to_tensor(img_as_img)
        
        #store bounding box information
        box = torch.tensor(list(zip( img_dat['x0'], img_dat['y0'], img_dat['x1'], img_dat['y1']))[0], dtype=torch.float32)
                
        scale = torch.tensor( [ 224 / img_dat['width'][0] , 224/ img_dat['height'][0] ,
                                224 / img_dat['width'][0] , 224/ img_dat['height'][0]])
        scaled_box = box * scale
        #normalized_box = normalize_boxes(box.unsqueeze(0), img_dat['width'][0], img_dat['height'][0])
        
        return img_as_tensor , scaled_box #, create_mask(box, img_as_tensor)

    def __len__(self) -> int:
        
        return self.data_len