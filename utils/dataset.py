"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Implement COCO Fork dataset on PyTorch
"""

from torch.utils.data import Dataset
import json
import cv2 as cv
from torch import as_tensor, float64

class CocoForkDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None, mode='train', device='cpu'):
        """
        Initiliaze the dataset
        
        Args
            - img_dir: Path to images
            - ann_dir: Path to annotations
            - transforms: Transformations to apply to image
            - mode: Mode of dataset ('train' or 'val')
            - device: Device to load the tensor
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.mode = mode
        self.device = device

        self.imgs = self.read_ids(self.ann_dir, self.mode)
        self.ann = self.read_json(self.ann_dir, self.mode)

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Get an item from dataset by index
        """
        # Load image and annotation
        img = cv.imread(f'{self.img_dir}/{str(self.imgs[idx]).zfill(12)}.jpg')
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        ann_list = [k for k in self.ann if self.imgs[idx] == k['image_id']]
        
        # Set ann as a dict
        ann = {}
        # ann['segmentation'] = [k['segmentation'] for k in ann_list]
        ann['area'] = [k['area'] for k in ann_list]
        # ann['iscrowd'] = [k['iscrowd'] for k in ann_list]
        ann['image_id'] = [k['image_id'] for k in ann_list]
        ann['boxes'] = [k['bbox'] for k in ann_list]
        ann['category_id'] = [k['category_id'] for k in ann_list]
        ann['id'] = [k['id'] for k in ann_list]

        boxes = []
        areas = []
        for box in ann['boxes']:
            xmin = float(box[0])
            ymin = float(box[1])
            xmax = float(box[0] + box[2])
            ymax = float(box[1] + box[3])
            boxes.append([xmin, ymin, xmax, ymax])
            area = (ymax - ymin) * (xmax - xmin)
            areas.append(float(area))

        ann['boxes'] = as_tensor(boxes, dtype=float64, device=self.device)
        ann['area'] = as_tensor(areas, dtype=float64, device=self.device)

        # Apply transformations
        if self.transforms is not None:
            img, ann = self.transforms(img, ann)

        img = img.to(self.device)

        return img, ann
    
    def read_json(self, ann_dir, mode):
        """
        Read json annotations file
        """
        input_json = f'{ann_dir}/instances_{mode}2017_filtered.json'
        with open(input_json) as json_file:
            coco = json.load(json_file)
        return coco['annotations']
    
    def read_ids(self, ann_dir, mode):
        """
        Read images IDs from txt file
        """
        input_ids = f'{ann_dir}/instances_{mode}2017_id.txt'
        with open(input_ids, 'r') as id_list:
            img_ids = [int(str(line[0:-1])) for line in id_list.readlines()]
        # Remove duplicates
        img_ids = list(dict.fromkeys(img_ids))
        return img_ids

# def show_image(img, bbox):
#     """
#     Draw the bbox and show the image
#     """
#     import numpy as np
#     img = np.float32(img)
#     img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
#     for box in bbox:
#         x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3]) 
#         cv.rectangle(img, (x0, y0), (x1, y1), (255,0,0), 2)
#     cv.imshow(f'IMG', img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# from utils import get_transforms

# dataset = CocoForkDataset('../data/train', '../data/annotations', get_transforms(train=True, size=(640, 640)), 'train')
# for k in range(20, 30):
#     img, ann = dataset.__getitem__(k)
#     # print(img.shape, ann)
#     show_image(img.numpy().transpose(1, 2, 0), ann['boxes'])