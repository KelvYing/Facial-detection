from cust_dataset import FaceDataset
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit 

def main() -> None:
    
    #train/ test sample split
    dat = pd.read_csv('./archive/faces.csv')
    
    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)
    split = splitter.split(dat, groups=dat['image_name'])
    train_inds, test_inds = next(split)

    train = dat.iloc[train_inds]
    test = dat.iloc[test_inds]


    train = FaceDataset(train, './archive/images/')
    test = FaceDataset(test, './archive/images/')
    print(len(train.image_arr))
    print(len(test.image_arr))

    boxes, img = next(iter(test))
    x, y, xx, yy = boxes
    print(x,y,xx,yy)
    
    boxes_tensor = torch.tensor(boxes).unsqueeze(0)
    img_bbox= draw_bounding_boxes(img , boxes_tensor, width = 3, colors = 'red')
    
    transform = transforms.Compose([
        transforms.ToPILImage()
    ])
    
    plt.imshow(transform(img_bbox))
    plt.show()
    print('here')



if __name__ == "__main__":
    main()