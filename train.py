from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit 

from cust_dataset import FaceDataset
from mask import create_mask

def main() -> None:
    
    #train/ test sample split
    dat = pd.read_csv('./archive/faces.csv')
    
    #split data into groups and making sure all bbox for an image are kept in train/test
    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)
    split = splitter.split(dat, groups=dat['image_name'])
    train_inds, test_inds = next(split)

    #slice the data
    train = dat.iloc[train_inds]
    test = dat.iloc[test_inds]

    #create the dataset objects
    train = FaceDataset(train, './archive/images/')
    test = FaceDataset(test, './archive/images/')
    print(train.data_len)
    print(test.data_len)

    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToPILImage()
        
    ])
    
        
    for img , boxes , mask  in test:
        #print(boxes)
        
        img_bbox = draw_bounding_boxes(img , boxes, width = 3, colors = 'red')



        #mask = create_mask(boxes, img)
        print(mask)
        plt.imshow(transform(mask.unsqueeze(0)), cmap='gray') #have to transform mask.unsqueeze(0) due to bad indices
        
        plt.show()
        plt.clf()
        plt.imshow(transform(img_bbox))
        plt.show()

    


    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one (number of classes + background)
    num_classes = 2  # Adjust this according to your dataset (number of object classes + background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


if __name__ == "__main__":
    main()