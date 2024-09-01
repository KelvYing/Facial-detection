from torchvision.utils import draw_bounding_boxes 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit 
from torchsummary import summary
import numpy as np
from PIL import Image

from cust_dataset import FaceDataset
from mask import create_mask
from models import RCNN

from ultralytics import YOLO

def calculate_iou(boxes1, boxes2):
    # Calculate IoU between predicted and target boxes
    boxes1 = boxes1.detach().cpu().numpy()
    boxes2 = boxes2.detach().cpu().numpy()
    
    x1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3], boxes2[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection
    
    #calculate center
    center1 = (boxes1[:, :2] + boxes1[:, 2:]) /2
    center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2
    
    #Calculate Euclidean distance
    center_distance = np.sqrt(np.sum((center1 - center2) ** 2, axis=1))
    
    # Normalize center distance by the diagonal of the image (assuming image size is 224 x 224)
    normalized_center_distance = center_distance / np.sqrt( (224 ** 2 + 224 ** 2) )

    
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    
    return torch.from_numpy(np.array(1 - np.mean(iou) + np.mean(normalized_center_distance), dtype=np.float64)).float().requires_grad_(True)


def train_batch(dataloader, model, optimizer, criterion, device):
    model.train()
    for epoch in range(10):
        # loss?
        total_loss = 0
        print('epoch : ', epoch)
        for idx, (images, targets) in enumerate(dataloader):
            print(idx)

            images = torch.permute(images, (0 , 2 , 3 , 1))
                
            images = images.to(device)
            targets = targets.to(device)
            
            #pass the images into the model
            output = model(images)
            
            #Compute Loss
            loss = criterion(output, targets)
            total_loss += loss.item()
            #Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # add loss? and print loss per epoch?
        avg_loss = total_loss / len(dataloader)
        print('Loss for epoch :', str(avg_loss))

def validate_batch(dataloader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images , bbox in dataloader:
            
            images = torch.permute(images, (0 , 2 , 3 , 1))
            
            images = images.to(device)
            bbox = bbox.to(device)

            outputs = model(images)
            loss = criterion( outputs, bbox )
            total_loss += loss.item()
            
            # Store predictions and targets for further evaluation
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(bbox.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Convert predictions and targets to numpy arrays
    # all_predictions = np.array(all_predictions)
    # all_targets = np.array(all_targets)
    
    # # Calculate IoU (Intersection over Union) for bounding boxes
    # iou = calculate_iou(all_predictions, all_targets)
    
    return avg_loss

            
def view(test, transform):
    #view images with bounding boxes and masks
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
      
def view_pred(transform, device, model):
    tttt = Image.open('./archive/images/00000006.jpg')
    ttt = transform(tttt).unsqueeze(0).to(device)

    with torch.no_grad():  # Disable gradient computation
        output = model(ttt)
    
    output = output.squeeze().cpu().numpy()  
        
    print(output)
    print(output * 224)
    
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(tttt)

    # Create a Rectangle patch
    rect = patches.Rectangle((output[0], output[1]), output[2] - output[0], output[3] - output[1],
                             linewidth=2, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.axis('off')
    plt.show()
    
    return output, tttt
    
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

    #create the data transformation
    transform_data = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        
    #create the dataset objects
    train = FaceDataset(train, './archive/images/', transform = transform_data)
    test = FaceDataset(test, './archive/images/', transform = transform_data)
    batch_size = 512  
    train_dl = DataLoader(train, batch_size = batch_size, shuffle = True)
    test_dl = DataLoader(test, batch_size = batch_size)

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToPILImage()
        
    # ])
    # for img , boxes in train:
    #     # print(boxes.reshape(-1,4))
    #     # img_bbox = draw_bounding_boxes(torch.reshape(img, (3, 224, 224)) , boxes.reshape(-1,4), width = 3, colors = 'red')
    #     # plt.imshow(img_bbox)
    #     # plt.show()
        
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(torch.permute(img, (2, 0, 1)))

    #     # Create a Rectangle patch
    #     rect = patches.Rectangle((boxes[0], boxes[1]), boxes[2] - boxes[0], boxes[3] - boxes[1],
    #                             linewidth=2, edgecolor='r', facecolor='none')

    #     # Add the patch to the Axes
    #     ax.add_patch(rect)

    #     plt.axis('off')
    #     plt.show()
    #     # print(boxes)

    
    #Create models and other stuff
    
    # model = RCNN().to(device)
    # print(summary(model, (3, 224, 224)))

    # optimizer = optim.Adam(model.parameters(), lr= 0.01)
    
    # #train the model
    # train_batch(train_dl, model, optimizer, calculate_iou, device)
    # #validate the model
    # val = validate_batch(test_dl, model, calculate_iou, device)
    # print(val)

    # #view an example result
    # bbox , image = view_pred(transform_data,device,model)
    
    model = YOLO('yolov8n.pt')
    
    results = model.train( data)

if __name__ == "__main__":
    main()