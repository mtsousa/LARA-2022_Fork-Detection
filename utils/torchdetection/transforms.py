"""
Adapted from https://github.com/pytorch/vision/tree/main/references/detection
"""

from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F, InterpolationMode, transforms as T

import cv2 as cv
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target

class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_h, orig_w = F.get_dimensions(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device)
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                # target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                # Refresh target area
                target["area"][:] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])

                return image, target

class Resize(T.Resize):
    def forward(self, image, target=None):
        _, orig_height, orig_width = F.get_dimensions(image)

        new_width = self.size[0]
        new_height = self.size[1]

        image = F.resize(image, self.size, interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            target["area"][:] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])

        return image, target

class GaussianNoise(nn.Module):
    """
    Based on https://github.com/pytorch/vision/issues/6192#issuecomment-1164176231
    """
    def __init__(self, sigma=(0.1, 2.0)):
        super().__init__()
        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float):
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor, ann) -> Tensor:
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        out_img = img + sigma * torch.randn_like(img)
        return out_img, ann

class Buffer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img: Tensor, ann) -> Tensor:
        return img, ann

def rotate_bbox(bbox, angle, img_shape, interpolation=InterpolationMode.BILINEAR, expand=False, center=None, fill=255):
    mask = torch.zeros(img_shape).numpy().transpose(1, 2, 0)
    mask += 255
    mask = np.uint8(mask)

    x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle(mask, (x0, y0), (x1, y1), (0, 0, 0), -1)

    mask = torch.as_tensor(mask).reshape(img_shape)
    mask_rot = F.rotate(mask, angle, interpolation=interpolation, expand=expand, center=center, fill=fill)
    mask_rot = mask_rot.numpy().transpose(1, 2, 0)
    mask_rot = np.float32(mask_rot)

    dst = cv.cornerHarris(mask_rot, 5, 3, 0.04)
    _, dst = cv.threshold(dst, 0.1*dst.max(), 255, 0)
    dst = np.uint8(dst)
    _, _, _, centroids = cv.connectedComponentsWithStats(dst)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(mask_rot, np.float32(centroids), (5, 5), (-1, -1), criteria)

    corners = [corners[k] for k in range(1, len(corners))]
    x_ = [k[0] for k in corners]
    y_ = [k[1] for k in corners]

    xmin = int(np.min(x_))
    ymin = int(np.min(y_))
    xmax = int(np.max(x_))
    ymax = int(np.max(y_))
    return torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float64)

class RandomRotation(T.RandomRotation):
    def __init__(self, degrees, interpolation=InterpolationMode.BILINEAR, expand=False, center=None, fill=0):
        super().__init__(degrees)
        self.degrees = degrees
        self.center = center
        self.interpolation = interpolation
        self.expand = expand
        self.fill = fill

    def forward(self, img, target):
        fill = self.fill
        channels, w, h = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        img = F.rotate(img, angle, self.interpolation, self.expand, self.center, fill)
        
        for box in range(len(target['boxes'])):
            target['boxes'][box][:] = rotate_bbox(target['boxes'][box][:], angle, img_shape=(1, w, h),
                                                  interpolation=self.interpolation, expand=self.expand, center=self.center, fill=255)
        return img, target

class RandomChoice(T.RandomChoice):
    # def __call__(self, *args):
    #     t = random.choices(self.transforms, weights=self.p)[0]
    #     print(t)
    #     return t(*args)
    pass