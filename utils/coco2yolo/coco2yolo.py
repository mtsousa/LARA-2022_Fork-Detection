"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Convert COCO annotations to YOLOv7 format
"""

import json
import cv2 as cv

def coco2yolo(img_dir, ann_dir, mode='train'):
    """
    Convert from COCO to YOLO
    """
    imgs = read_ids(ann_dir, mode)
    ann = read_json(ann_dir, mode)

    for idx in range(len(imgs)):
        # Load image and annotation
        h, w, _ = cv.imread(f'{img_dir}/{str(imgs[idx]).zfill(12)}.jpg').shape
        img_name = f'{img_dir}/{str(imgs[idx]).zfill(12)}.txt'
        ann_list = [k for k in ann if imgs[idx] == k['image_id']]
        
        # Set ann as a dict
        ann_dict = {}
        ann_dict['boxes'] = [k['bbox'] for k in ann_list]
        ann_dict['labels'] = [0 for k in ann_list]

        boxes = []
        for box, label in zip(ann_dict['boxes'], ann_dict['labels']):
            xmin = float(box[0])
            ymin = float(box[1])
            xmax = float(box[0] + box[2])
            ymax = float(box[1] + box[3])

            # Transform the bbox co-ordinates as per the format required by YOLO v5
            b_center_x = (xmin + xmax)/2 
            b_center_y = (ymin + ymax)/2
            b_width    = (xmax - xmin)
            b_height   = (ymax - ymin)
            
            # Normalise the co-ordinates by the dimensions of the image
            b_center_x /= w 
            b_center_y /= h 
            b_width    /= w 
            b_height   /= h 

            boxes.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(label, b_center_x, b_center_y, b_width, b_height))
        
        print("\n".join(boxes), file=open(img_name, "w"))

def read_json(ann_dir, mode):
    """
    Read json annotations file
    """
    input_json = f'{ann_dir}/instances_{mode}2017_filtered.json'
    with open(input_json) as json_file:
        coco = json.load(json_file)
    return coco['annotations']

def read_ids(ann_dir, mode):
    """
    Read images IDs from txt file
    """
    input_ids = f'{ann_dir}/instances_{mode}2017_id.txt'
    with open(input_ids, 'r') as id_list:
        img_ids = [int(str(line[0:-1])) for line in id_list.readlines()]
    # Remove duplicates
    img_ids = list(dict.fromkeys(img_ids))
    return img_ids

coco2yolo('../../data/train', '../../data/annotations', 'train')
coco2yolo('../../data/val', '../../data/annotations', 'val')