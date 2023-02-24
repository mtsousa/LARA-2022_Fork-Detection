"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Show dataset images with ground truth bounding box from COCO dataset annotations
"""

import json
import cv2 as cv
import argparse
from random import shuffle

import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[2]
print(package_root_directory)
sys.path.append(str(package_root_directory))

from utils.utils import plot_one_box, show_predicted_image

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
    # Remove duplicates
    img_ids = list(dict.fromkeys(img_ids))
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
        image = images[k]['name']
        img = cv.imread(f'../../data/{dataset}/{image}')

        for j in images[k]['bbox']:
            x, y, w, h = int(j[0]), int(j[1]), int(j[2]), int(j[3])
            box = [x, y, x+w, y+h]
            plot_one_box(box, img, label='fork', color=[104, 184, 82], line_thickness=1)
        
        show_predicted_image(img)
