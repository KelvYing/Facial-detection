from cust_dataset import CelebDataset

def main() -> None:
    test = CelebDataset('archive/list_bbox_celeba.csv',
                        'archive/list_attr_celeba.csv',
                        'archive/list_landmarks_align_celeba.csv',
                        'archive/list_eval_partition.csv',
                        './archive/img_align_celeba/img_align_celeba/')

    # for i, (boxes) in enumerate(test):
    #     x, y, xx, yy = boxes
    #     print(x,y,xx,yy)
        
        
    (boxes) = next(iter(test))
    x, y, xx, yy = boxes
    print('here')
    print(x,y,xx,yy)
    




if __name__ == "__main__":
    main()