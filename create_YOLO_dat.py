import pandas as pd
import os
from ultralytics import YOLO
import torch
import torchvision

def main():
    csv_tot = pd.read_csv('faces.csv')
    for name in csv_tot['image_name']:
        temp_dat = csv_tot.loc[csv_tot['image_name'] == name, :].copy()
        
        temp_dat['class'] = 0
        
        temp_dat['width_box'] = (temp_dat['x1'] - temp_dat['x0'])
        temp_dat['height_box'] = (temp_dat['y1'] - temp_dat['y0'])
        
        temp_dat['x_center'] = (temp_dat['x0'] + temp_dat['width_box'] / 2) / temp_dat['width']
        temp_dat['y_center'] = (temp_dat['y0'] + temp_dat['height_box'] / 2) / temp_dat['height']
        
        temp_dat['width_box'] = temp_dat['width_box'] / temp_dat['width']
        temp_dat['height_box'] = temp_dat['height_box'] / temp_dat['height']
        
        temp_dat = temp_dat[['class', 'x_center', 'y_center', 'width_box', 'height_box']]
        temp_dat.to_csv('archive/labels/' + name.replace('jpg', 'txt'), sep= ' ', header = False, index = False)

def runYOLO():
    print(torch.__version__)
    model = YOLO('yolov8n.pt')
    model = model.to('cuda:0')
    results = model.train(
        data='config.yaml',
        epochs= 50,
        batch=8,
    )
    results = model.val()

    results = model('archive/images/00000004.jpg')

if __name__ == '__main__':
    main()
    runYOLO()