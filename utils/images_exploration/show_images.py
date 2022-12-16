"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Show dataset images with ground truth bounding box from COCO dataset annotations
"""

import json
import cv2 as cv
import argparse
from random import shuffle

def read_json(input_name):
    """
    Read json annotations file
    """
    input_json = '../../data/annotations/' + input_name + '_filtered.json'
    with open(input_json) as json_file:
        coco = json.load(json_file)
    return coco

def read_ids(input_name):
    """
    Read images IDs from txt file
    """
    input_ids = '../../data/annotations/' + input_name + '_id.txt'
    with open(input_ids, 'r') as id_list:
        img_ids = [int(str(line[0:-1])) for line in id_list.readlines()]
    shuffle(img_ids)
    return img_ids

def find_image(coco, img_ids):
    """
    Find image annotations and return the file name and the bbox
    """
    images = dict()
    for image in coco['images']:
        for ann in coco['annotations']:
            image_id = image['id']
            ann_id = ann['image_id']
            if image_id in img_ids and image_id == ann_id:
                name = image['file_name']
                bbox = ann['bbox']
                if image_id not in list(images.keys()):
                    images[image_id] = {'name': name, 'bbox': [bbox]}
                else:
                    images[image_id]['bbox'].append(bbox)
    return images

def show_image(id, dataset, file_name, bbox):
    """
    Draw the bbox and show the image
    """
    img = cv.imread(f'../../data/{dataset}/{file_name}')
    # "bbox": [x,y,width,height]
    for box in bbox:
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
    cv.imshow(f'IMG {id}', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Explore COCO dataset.')
    parser.add_argument('-d', '--dataset',
                        help='Type of dataset \'train\' or \'val\'')
    parser.add_argument('-n', '--num',
                        help='Number of images')

    args = parser.parse_args()
    dataset = args.dataset
    annotations_name = f'instances_{dataset}2017'
    n = int(args.num)

    # Load the json
    print('Loading json file...')
    coco = read_json(annotations_name)

    # Find n images to show
    print('Searching for images...')
    img_ids = read_ids(annotations_name)
    images = find_image(coco, img_ids[:n])

    # For each image, show the bbox
    print('Loading images...')
    for k in images.keys():
        show_image(k, dataset, images[k]['name'], images[k]['bbox'])