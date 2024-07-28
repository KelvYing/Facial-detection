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
    
    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)
    split = splitter.split(dat, groups=dat['image_name'])
    train_inds, test_inds = next(split)

    train = dat.iloc[train_inds]
    test = dat.iloc[test_inds]


    train = FaceDataset(train, './archive/images/')
    test = FaceDataset(test, './archive/images/')
    print(train.data_len)
    print(test.data_len)

    #boxes, img = next(iter(test))

    for img , boxes  in test:
        #print(boxes)
        
        img_bbox= draw_bounding_boxes(img , boxes, width = 3, colors = 'red')

        transform = transforms.Compose([
            transforms.ToPILImage()
        ])

        mask = create_mask(boxes, img)
        plt.imshow(mask, cmap='gray')
        #plt.imshow(transform(img_bbox))
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