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
    if train:
        random_transforms = []
        random_transforms.append(T.RandomHorizontalFlip(0.5))
        random_transforms.append(T.RandomIoUCrop())
        random_transforms.append(T.GaussianNoise(sigma=(0.1, 0.15)))
        random_transforms.append(T.Buffer())
        # random_transforms.append(T.RandomRotation(degrees=(-20, 20)))
        p = [1/len(random_transforms) for k in range(len(random_transforms))]
        transforms.append(T.RandomChoice(random_transforms, p))
    transforms.append(T.Resize((size)))
    return T.Compose(transforms)