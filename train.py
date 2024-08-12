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

from cust_dataset import FaceDataset
from mask import create_mask
from models import RCNN

def train_batch(dataloader, model, optimizer, criterion, device):
    model.train()
    for epoch in range(5):
        # loss?
        print('epoch : ', epoch)
        for batch_idx, ( images, bbox ) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            
            #pass the images into the model
            output = model(images)
            
            #Compute Loss
            loss = criterion(output, targets)
            
            #Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # add loss? and print loss per epoch?
    
def validate_batch(model, criterion, dataloader, device):
    model.eval()
    with torch.no_grad():
        for images , bbox in dataloader:
            images = images.to(device)
            bbox = bbox.to(device)
            outputs = model(images)
            loss = criterion( outputs, bbox )
        
def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    batch_size = 32
    train_dl = DataLoader(train, batch_size = batch_size, shuffle = True)
    test_dl = DataLoader(test, batch_size = batch_size)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # Get the number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

    model = RCNN().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr= 0.002)



if __name__ == "__main__":
    main()