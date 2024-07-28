import numpy as np
import torch

def create_mask(bbox, pic) -> torch.Tensor:
    rows, cols, _ = pic.shape
    res = torch.zeros((rows,cols))
    
    for bb in bbox:
        print(bb)
        res[bb[0] : bb[2] , bb[1] : bb[3]] = 1
        
    return res