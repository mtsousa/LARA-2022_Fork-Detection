"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Implement some transformations to object detection with PyTorch

Adapted from https://github.com/pytorch/vision/blob/main/references/detection/transforms.py
"""

from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F, InterpolationMode, transforms as T

import cv2 as cv
import numpy as np

class Buffer(nn.Module):
    """
    Return the original image
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, img: Tensor, target: Dict):
        return img, target

class Compose:
    """
    Composes several transforms together
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ConvertImageDtype(nn.Module):
    """
    Convert a tensor image to the given dtype and scale the values accordingly
    This function does not support PIL Image
    
    Args
        - dtype (torch.dtype): Desired data type of the output
    """
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, image: Tensor, target: Dict):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class GaussianBlur(T.GaussianBlur):
    """
    Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.
    
    Args
        - kernel_size (int or sequence): Size of the Gaussian kernel
        - sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring
    """
    def forward(self, img: Tensor, target: Dict):
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return F.gaussian_blur(img, self.kernel_size, [sigma, sigma]), target

class GaussianNoise(nn.Module):
    """
    Randomly add Gaussian noise to a given imagem.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    
    Args
        - sigma (float or tuple of float (min, max)): Standard deviation of the noise
    
    Based on https://github.com/pytorch/vision/issues/6192#issuecomment-1164176231
    """
    def __init__(self, sigma: Tuple[float, float] = (0.1, 2.0)):
        super().__init__()
        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float):
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor, target: Dict):
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        out_img = img + sigma * torch.randn_like(img)
        return out_img, target

class PILToTensor(nn.Module):
    """
    Convert a PIL Image to a tensor of the same type.
    Convert a PIL Image (H x W x C) to a Tensor of shape (C x H x W)
    """
    def forward(self, image: Tensor, target: Dict):
        image = F.pil_to_tensor(image)
        return image, target

class RandomChoice(T.RandomChoice):
    """
    Apply single transformation randomly picked from a list
    """
    pass

class RandomGrayscale(T.RandomGrayscale):
    """
    Randomly convert image to grayscale with a probability of p (default 0.1).
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions
    
    Args
        - p (float): Probability that image should be converted to grayscale
    """
    def forward(self, img: Tensor, target: Dict):
        num_output_channels, _, _ = F.get_dimensions(img)
        if torch.rand(1) < self.p:
            return F.rgb_to_grayscale(img, num_output_channels=num_output_channels), target
        return img, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    
    Args
        - p (float): Probability of the image being flipped. Default value is 0.5
    """
    def forward(self, image: Tensor, target: Dict):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target

class RandomIoUCrop(nn.Module):
    """
    Randomly crop a given image in the ROI

    Args
        - min_scale: Minimum value for scale
        - max_scale: Maximum value for scale
        - min_aspect_ratio: Minimum value for output image aspect ratio
        - max_aspect_ratio: Maximum value for output image aspect ratio
        - sampler_options: List of values for Jaccard index
        - trials: Number of trials of crop
    """
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

    def forward(self, image: Tensor, target: Dict):
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
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                # Refresh target area
                target["area"] = torch.zeros(len(target["boxes"]))
                target["area"][:] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])

                return image, target

class RandomPerspective(T.RandomPerspective):
    """
    Performs a random perspective transformation of the given image with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    
    Args
        - distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1. Default is 0.5
        - p (float): probability of the image being transformed. Default is 0.5
        - interpolation (InterpolationMode): Desired interpolation
        - fill (sequence or number): Pixel fill value for the area outside the transformed image. Default is 0
    """
    def forward(self, img: Tensor, target: Dict):
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        bbox = target['boxes'].detach().clone()

        if torch.rand(1) < self.p:
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            img_p = F.perspective(img, startpoints, endpoints, self.interpolation, fill)
            
            min_area = []
            for box in range(len(target['boxes'])):
                target['boxes'][box][:], flag = self.perspective_bbox(target['boxes'][box][:], startpoints, endpoints,
                                                                      img_shape=(1, height, width))
                min_area.append(flag)
            
            if not any(min_area):
                target['boxes'] = bbox
                return img, target

            else:
                # Remove min area
                target['boxes'] = target["boxes"][min_area]
                
                # Refresh target area
                target["area"] = torch.zeros(len(target["boxes"]))
                target["area"][:] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
                
                return img_p, target
        return img, target

    def perspective_bbox(self, bbox, startpoints, endpoints, img_shape, fill=255):
        # Create a mask and add the bounding box
        mask = torch.zeros(img_shape).numpy().transpose(1, 2, 0)
        mask += 255
        mask = np.uint8(mask)

        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        area_orig = (x1-x0)*(y1-y0)
        cv.rectangle(mask, (x0, y0), (x1, y1), (0, 0, 0), -1)

        # Apply perspective transformation to the mask
        mask = torch.as_tensor(mask).reshape(img_shape)
        mask_persp = F.perspective(mask, startpoints, endpoints, self.interpolation, fill)
        mask_persp = mask_persp.numpy().transpose(1, 2, 0)
        mask_persp = np.float32(mask_persp)

        # Find the corners of bounding box on the transformed mask
        dst = cv.cornerHarris(mask_persp, 5, 3, 0.04)
        _, dst = cv.threshold(dst, 0.1*dst.max(), 255, 0)
        dst = np.uint8(dst)
        _, _, _, centroids = cv.connectedComponentsWithStats(dst)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv.cornerSubPix(mask_persp, np.float32(centroids), (5, 5), (-1, -1), criteria)

        # Extract the corners of the rectangle parallel to image bottom 
        corners = [corners[k] for k in range(1, len(corners))]
        x_ = [k[0] for k in corners]
        y_ = [k[1] for k in corners]

        xmin = int(np.min(x_))
        ymin = int(np.min(y_))
        xmax = int(np.max(x_))
        ymax = int(np.max(y_))

        # Warn if the new area is less than 20% of the original
        area_persp = (xmax-xmin)*(ymax-ymin)
        flag = True
        if area_persp < 0.2*area_orig:
            flag = False

        return torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float64), flag

class RandomRotation(T.RandomRotation):
    """
    Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    
    Args
        - degrees (tuple of float): Range of degrees to select from (min, max)
        - interpolation (InterpolationMode): Desired interpolation
        - expand (bool, optional): Optional expansion flag
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
        - center (sequence, optional): Optional center of rotation, (x, y). Default is the center of the image
        - fill (sequence or number): Pixel fill value for the area outside the rotated image. Default is 0
    """
    def __init__(self, degrees: Tuple[float, float], interpolation=InterpolationMode.BILINEAR,
                 expand=False, center=None, fill=0):
        super().__init__(degrees)
        self.degrees = degrees
        self.center = center
        self.interpolation = interpolation
        self.expand = expand
        self.fill = fill

    def forward(self, img: Tensor, target: Dict):
        fill = self.fill
        channels, h, w = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        img = F.rotate(img, angle, self.interpolation, self.expand, self.center, fill)
        
        for box in range(len(target['boxes'])):
            target['boxes'][box][:] = self.rotate_bbox(target['boxes'][box][:], angle, img_shape=(1, h, w))

        # Refresh target area
        target["area"] = torch.zeros(len(target["boxes"]))
        target["area"][:] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])

        return img, target

    def rotate_bbox(self, bbox, angle, img_shape, fill=255):
        # Create a mask and add the bounding box
        mask = torch.zeros(img_shape).numpy().transpose(1, 2, 0)
        mask += 255
        mask = np.uint8(mask)

        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv.rectangle(mask, (x0, y0), (x1, y1), (0, 0, 0), -1)

        # Rotate the mask by the given angle
        mask = torch.as_tensor(mask).reshape(img_shape)
        mask_rot = F.rotate(mask, angle, interpolation=self.interpolation, expand=self.expand, center=self.center, fill=fill)
        mask_rot = mask_rot.numpy().transpose(1, 2, 0)
        mask_rot = np.float32(mask_rot)

        # Find the corners of bounding box on the rotated mask
        dst = cv.cornerHarris(mask_rot, 5, 3, 0.04)
        _, dst = cv.threshold(dst, 0.1*dst.max(), 255, 0)
        dst = np.uint8(dst)
        _, _, _, centroids = cv.connectedComponentsWithStats(dst)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv.cornerSubPix(mask_rot, np.float32(centroids), (5, 5), (-1, -1), criteria)

        # Extract the corners of the rectangle parallel to image bottom 
        corners = [corners[k] for k in range(1, len(corners))]
        x_ = [k[0] for k in corners]
        y_ = [k[1] for k in corners]

        xmin = int(np.min(x_))
        ymin = int(np.min(y_))
        xmax = int(np.max(x_))
        ymax = int(np.max(y_))
        return torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float64)

class RandomTranslation(nn.Module):
    """
    Randomly translate the image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    
    Args
        - fill (sequence or number): Pixel fill value for the area outside the rotated image. Default is 0
        - side_range (tuple of float): Range of output image side (min, max)
        - p (float): Probability of the image being translated. Default value is 0.5
        - sampler_options (list): List of values for Jaccard index
    """
    def __init__(self, fill: float = 0, side_range: Tuple[float, float] = (1.0, 4.0),
                 p: float = 0.5, sampler_options: Optional[List[float]] = None):
        super().__init__()
        self.padding = RandomZoomOut(fill=fill, side_range=side_range, p=p)
        self.trials = 40
        if sampler_options is None:
            sampler_options = [0.3, 0.5, 0.7, 0.9]
        self.options = sampler_options

    def forward(self, img: Tensor, target: Dict):
        _, h, w = F.get_dimensions(img)
        self.crop_img = T.CenterCrop(size=(h, w))
        box = target['boxes'].detach().clone()
        target_box = {}
        target_box['boxes'] = target['boxes'].detach().clone()
        
        for _ in range(self.trials):
            img_pad, target_pad = self.padding(img.detach().clone(), target_box)
            hc, wc = h, w
            _, hp, wp = F.get_dimensions(img_pad)
            right = int(wp/2 + w/2)
            bottom = int(hp/2 + h/2)
            left = int(wp/2 - w/2)
            top = int(hp/2 - h/2)

            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]

            # check for any valid boxes with centers within the crop area
            cx = 0.5 * (target_pad["boxes"][:, 0] + target_pad["boxes"][:, 2])
            cy = 0.5 * (target_pad["boxes"][:, 1] + target_pad["boxes"][:, 3])
            is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
            if is_within_crop_area.any():
                # check at least 1 box with jaccard limitations
                boxes = target_pad["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device)
                )
                if ious.max() < min_jaccard_overlap:
                    # keep only valid boxes and perform cropping
                    target["boxes"] = boxes
                    target["boxes"][:, 0::2] -= left
                    target["boxes"][:, 1::2] -= top
                    target["boxes"][:, 0::2].clamp_(min=0, max=wc)
                    target["boxes"][:, 1::2].clamp_(min=0, max=hc)

                    image = self.crop_img(img_pad)

                    # Refresh target area
                    target["area"] = torch.zeros(len(target["boxes"]))
                    target["area"][:] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])

                    return image, target

            target_box['boxes'] = target['boxes'].detach().clone()
            
        target['boxes'] = box
        return img, target

class RandomVerticalFlip(T.RandomVerticalFlip):
    """
    Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    
    Args
        - p (float): Probability of the image being flipped. Default value is 0.5
    """
    def forward(self, image: Tensor, target: Dict):
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target is not None:
                _, height, _ = F.get_dimensions(image)
                target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
        return image, target

class RandomZoomOut(nn.Module):
    """
    Randomly zoom out an image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    
    Args
        - fill (sequence or number): Pixel fill value for the area outside the rotated image. Default is 0
        - side_range (tuple of float): Range of output image side (min, max)
        - p (float): Probability of the image being translated. Default value is 0.5
    """
    def __init__(self, fill: Optional[List[float]] = None,
                 side_range: Tuple[float, float] = (1.0, 4.0), p: float = 0.5):
        super().__init__()
        if fill is None:
            fill = [0.0, 0.0, 0.0]
        self.fill = fill
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid canvas side range provided {side_range}.")
        self.p = p

    def forward(self, image: Tensor, target: Dict):
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        if torch.rand(1) >= self.p:
            return image, target

        _, orig_h, orig_w = F.get_dimensions(image)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)
        fill = 0

        image = F.pad(image, [left, top, right, bottom], fill=fill)
        if isinstance(image, torch.Tensor):
            # PyTorch's pad supports only integers on fill. So we need to overwrite the colour
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., (top + orig_h) :, :] = image[
                ..., :, (left + orig_w) :
            ] = v

        if target is not None:
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top
        
            # Refresh target area
            target["area"] = torch.zeros(len(target["boxes"]))
            target["area"][:] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])

        return image, target

class Resize(T.Resize):
    """
    Resize the input image to the given size.
    If the image is torch Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions
    """
    def forward(self, image: Tensor, target: Dict):
        _, orig_height, orig_width = F.get_dimensions(image)

        new_width = self.size[0]
        new_height = self.size[1]

        image = F.resize(image, self.size, interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            target["area"][:] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])

        return image, target

class ToTensor(object):
    """
    Convert a image to a tensor of the same type
    """
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target