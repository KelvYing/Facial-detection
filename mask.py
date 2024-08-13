import numpy as np
import torch

def create_mask(bbox, pic) -> torch.Tensor:
    print(pic.shape)
    _, rows, cols, = pic.shape
    res = torch.zeros((rows,cols))
    
    for bb in bbox:
        res[int(bb.data[1].item()) : int(bb.data[3].item()) , int(bb.data[0].item()) : int(bb.data[2].item())] = 1
        
    return res