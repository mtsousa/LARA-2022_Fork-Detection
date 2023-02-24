"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Implement some functions to model and dataset
"""

from .torchdetection import transforms as T
import cv2 as cv
import numpy as np
from random import randint
from torch import tensor, float64
from math import ceil

def get_transforms(train, size):
    """
    Get transformations to apply to dataset

    Args
        - train: Bool to add the transformation to data augmentation
        - size: Size of output image
    """
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.ConvertImageDtype(float))
    if train:
        random_transforms = []
        random_transforms.append(T.Buffer())
        random_transforms.append(T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)))
        random_transforms.append(T.GaussianNoise(sigma=(0.1, 0.15)))
        random_transforms.append(T.RandomGrayscale(p=1.0))
        random_transforms.append(T.RandomHorizontalFlip(p=1.0))
        random_transforms.append(T.RandomIoUCrop(sampler_options=[0.3, 0.5, 0.7, 0.9]))
        random_transforms.append(T.RandomPerspective(p=1.0))
        random_transforms.append(T.RandomRotation(degrees=(-20, 20), expand=True))
        random_transforms.append(T.RandomTranslation(p=1.0))
        random_transforms.append(T.RandomVerticalFlip(p=1.0))
        random_transforms.append(T.RandomZoomOut(side_range=(1, 1.5), p=1.0))
        p = [1/len(random_transforms) for k in range(len(random_transforms))]
        transforms.append(T.RandomChoice(random_transforms, p))
    transforms.append(T.Resize((size)))
    return T.Compose(transforms)

def show_predicted_image(img):
    """
    Show the predicted image
    """
    cv.imshow('Predicted image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    """
    Resize and pad image while meeting strid-multiple constraints

    Args
        - im: Image to resize
        - new_shape: Shape of resized image
        - color: Padding color
        - auto: Flag to autoscale
        - scaleup: Flag to scale up the image
        - stride: Stride value

    Adapted from https://github.com/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb
    """
    shape = im.shape[:2]  # Current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # Only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # Minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # Divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # Resize
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # Add border
    return im, r, (dw, dh)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    Plot one bounding box on image
    
    Args
        - x: Bounding box coordinates
        - img: Image
        - color: Color of bounding box
        - label: Class label
        - line_tickness: Bounding box line thickness
    
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # Line/font thickness
    color = color or [randint(0, 255) for _ in range(3)]

    # Plot rectangle
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv.rectangle(img, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)
    
    # Plot label
    if label:
        tf = max(tl - 1, 1)  # Font thickness
        t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv.rectangle(img, c1, c2, color, -1, cv.LINE_AA)  # Filled
        cv.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv.LINE_AA)

def adjust_image(src_image, shape):
    """
    Adjust image shape to inference

    Args
        - src_image: Image path
        - shape: New shape
    
    """
    img = cv.imread(src_image)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    image = img.copy()

    # Adjust image shape
    image, ratio, dwdh = letterbox(image, new_shape=shape, auto=False)
    
    # Transpose and set as a contiguous array
    image = image.transpose((2, 0, 1))
    image = np.ascontiguousarray(image)
    image = np.expand_dims(image, 0) # Expand image dimensions
    im = image.astype(np.float32)

    # Extract image name
    name = src_image.split('/')[-1]

    return {"image": im, "ratio": tensor([ratio], dtype=float64),"dwdh": dwdh, "name": [name]}

def plot_mosaic(images, paths=None, fname='images.jpg', max_size=640, max_subplots=16):
    """
    Plot image grid with labels

    Args
        - images: Batch of images to create the grid
        - paths: Path to the image names
        - fname: Output file name
        - max_size: Image maximum size
        - max_subplots: Maximum number of subplots
    
    Adapted from https://github.com/WongKinYiu/yolov7/blob/13594cf6d42bc3a49ff226570aa77b6bf7615f7f/utils/plots.py#L114
    """
    # Un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # Line thickness
    tf = max(tl - 1, 1)  # Font thickness
    bs, _, h, w = images.shape  # Batch size, _, height, width
    bs = min(bs, max_subplots)  # Limit plot images
    ns = np.ceil(bs ** 0.5)  # Number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = ceil(scale_factor * h)
        w = ceil(scale_factor * w)

    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # Init
    for i, img in enumerate(images):
        if i == max_subplots:  # If last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img

        # Draw image filename labels
        if paths:
            label = paths[i][:40] # Trim to 40 char
            t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv.LINE_AA)

        # Image border
        cv.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=2)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # Ratio to limit image size
        mosaic = cv.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv.INTER_AREA)
        cv.imwrite(fname, cv.cvtColor(mosaic, cv.COLOR_BGR2RGB))
    
    return mosaic
