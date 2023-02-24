"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Implement COCO Fork dataset on PyTorch
"""

from torch.utils.data import Dataset
import json
import cv2 as cv
from torch import as_tensor, float64
from .utils import letterbox
import numpy as np
import os

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
        # As this dataset only has one class, this class has the number 1 as label
        ann['labels'] = as_tensor([1 for k in ann_list])

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

class TestDataset(Dataset):
	def __init__(self, path, shape):
		"""
        Initiliaze the dataset
        
        Args
            - path: Path to images
            - shape: Input shape
        """
		self.path = path
		self.imgs_list = os.listdir(path)
		self.shape = shape

	def __len__(self):
		"""
		Return the length of the dataset
		"""
		return len(self.imgs_list)

	def __getitem__(self, idx):
		"""
		Get an item from dataset by index with transformations
		"""
		img = cv.imread(self.path + "/" + self.imgs_list[idx])
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		image = img.copy()

        # Adjust image shape
		image, ratio, dwdh = letterbox(image, new_shape=self.shape, auto=False)
		
		# Transpose and set as a contiguous array
		image = image.transpose((2, 0, 1))
		image = np.ascontiguousarray(image)
		im = image.astype(np.float32)
        
		return {"image": im, "ratio": ratio,"dwdh": dwdh, "name": self.imgs_list[idx]}

	def __getsrc__(self, idx):
		"""
        Get an item from dataset by index without transformations
        """
		img = cv.imread(self.path + "/" + self.imgs_list[idx])
		return img
