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