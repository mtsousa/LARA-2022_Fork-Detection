"""
@Author: Matheus Teixeira de Sousa (mtsousa14@gmail.com)

Implement some functions to model and dataset
"""

from torchdetection import transforms as T

def get_transforms(train, size):
    """
    Get transformations to apply to dataset

    Args
        - train: Bool to add the transformation to data augmentation
        - size: Size of image to resize
    """
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.ConvertImageDtype(float))
    transforms.append(T.Resize((size)))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)