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
    for param in model.parameters():
        param.requires_grad = False
    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential (  nn.Linear(num_ftrs, 128),
                                nn.Linear(128, 64),
                                nn.Linear(64, 6),
                             )

    model = model.to(device)

    optimizer = optim.Adam(model.fc.parameters(), lr= 0.006)

    for i in range(5):

        for img, boxes, mask in train_dl:

            #transfer image tensor, box values and mask tensor to device
            img = img.to(device).float()
            boxes = boxes.to(device)
            mask = mask.to(device).float()

            #get output
            out_bb = model(img)
            #get loss for predicted mask tensor
            # loss_bb = F.l1_loss(out_bb, mask, reduction="none").sum(1)
            # loss_bb = loss_bb.sum()
            # loss = loss_class + loss_bb/C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            optimizer.zero_grad()
            outputs, output_mask = model(img, mask)
            loss = criterion(outputs, ...)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()