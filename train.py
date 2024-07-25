from cust_dataset import CelebDataset
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def main() -> None:
    test = CelebDataset('archive/faces.csv',
                        './archive/images/')
    #dat = pd.read_csv
    # for i, (boxes) in enumerate(test):
    #     x, y, xx, yy = boxes
    #     print(x,y,xx,yy)
        
        
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
    #print(x,y,xx,yy)
    




if __name__ == "__main__":
    main()