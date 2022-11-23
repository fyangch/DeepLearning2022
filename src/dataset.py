import os

import cv2
import PIL
from tqdm import tqdm

import numpy as np

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, RandomResizedCrop, RandomGrayscale, RandomHorizontalFlip, GaussianBlur, ColorJitter, RandomSolarize, ToPILImage, RandomCrop, CenterCrop, Resize

from typing import List, Union, Tuple


# transform that will be applied to every raw image
TINY_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# random augmentations
RELIC_AUGMENTATIONS = [
    RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    RandomGrayscale(p=0.5),
    GaussianBlur(kernel_size=23, sigma=(0.1, 0.2)),
    RandomSolarize(0.5, p=0.5),
]


def load_tiny_imagenet(
        tiny_imagenet_folder: str,
        dataset_type: str = 'train',
        transform=ToPILImage()
) -> List[PIL.Image]:
    """
    Helper function to load tiny-imagenet-200 dataset.

    Parameters
    ----------
    tiny_imagenet_folder
        Path to the 'tiny-imagenet-200' folder.
    dataset_type
        One of 'train', 'val', 'test' specifying which dataset to load.
    transform
        A torchvision transform that will be applied to every loaded image.

    Returns
    -------
    List[PIL.Image]
        A list of the loaded PIL images.
    """

    images = []
    if dataset_type == 'train':
        for foldername in tqdm(os.listdir(os.path.join(tiny_imagenet_folder, dataset_type))):
            for filename in os.listdir(os.path.join(tiny_imagenet_folder, dataset_type, foldername, "images")):
                img = cv2.imread(os.path.join(tiny_imagenet_folder, dataset_type, foldername, "images", filename))
                if img is not None:
                    images.append(transform(img) if transform else img)
    else:
        # 'val' or 'test'
        for filename in tqdm(os.listdir(os.path.join(tiny_imagenet_folder, dataset_type, "images"))):
            img = cv2.imread(os.path.join(tiny_imagenet_folder, dataset_type, "images", filename))
            if img is not None:
                images.append(transform(img) if transform else img)

    return images


def image_to_patches(img: torch.Tensor) -> List[torch.Tensor]:
    """
    Cuts an image into 9 patches and returns them in row major order.

    Parameters
    ----------
    img
        torch.Tensor image to be split into patches.

    Returns
    -------
    List[torch.Tensor]
        A list of the 9 patches.
    """
    splits_per_side = 3  # split of patches per image side
    img_size = img.size()[-1]
    grid_size = img_size // splits_per_side
    patch_size = img_size // 4

    # we first use a center crop (to ensure gap) followed by a random crop (jitter) followed by resize (to ensure img_size stays the same)
    random_jitter = Compose([CenterCrop(grid_size - patch_size // 4), RandomCrop(patch_size), Resize(img_size)])
    patches = [
        random_jitter(TF.crop(img, i * grid_size, j * grid_size, grid_size, grid_size))
        for i in range(splits_per_side)
        for j in range(splits_per_side)
    ]

    return patches


class OriginalPatchLocalizationDataset(torch.utils.data.Dataset):
    """
    Dataset implementing the original Patch Localization method
    A sample is made up of the 8 possible tasks for a given grid ((center, neighbor), labels)
    """

    def __init__(
            self,
            data_source: List[Union[PIL.Image, np.ndarray, torch.Tensor]],
            transform=None,
            samples_per_image: int = 8,
    ):
        """
        Parameters
        ----------
        data_source
            A list of images in one of these formats: PIL image, numpy ndarray, torch tensor.
        transform
            A torchvision transform that will be applied to every raw image.
        samples_per_image
            How many samples to take from each image. Has to be an integer between 1 and 8.
        """
        self.data_source = data_source
        self.transform = transform if transform else TINY_IMAGENET_TRANSFORM
        self.samples_per_image = samples_per_image

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        # load image from data_source
        img = self.data_source[idx]
        # apply transform
        img = self.transform(img)
        # get samples_per_image samples from img
        samples = self.image_to_samples(img)

        return samples

    def image_to_samples(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        img
            An image in torch.Tensor format.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            features, labels tuple
            features: torch.Tensor of shape (samples_per_image, 2, *img.size())
            labels: torch.Tensor of shape (samples_per_image) containing the corresponding labels (integers between 0 and 7)
        """
        # convert image into patches
        patches = image_to_patches(img)
        # randomly select n_samples from all possible labels without replacement
        labels = np.random.choice(8, self.samples_per_image)

        samples = []
        for label in labels:
            if label >= 4:
                # the middle patch (number 4) is never a neighbor
                label += 1
            samples.append(torch.stack((patches[4], patches[label])))

        return torch.stack(samples), torch.from_numpy(labels)


class OurPatchLocalizationDataset(OriginalPatchLocalizationDataset):
    """
    Dataset implementing our modified Patch Localization method
    A sample is made up of the 8 possible tasks for a given grid ((center, random_aug(neighbor), random_aug(neighbor)), labels)
    """

    def __init__(self, data_source, transform=None, samples_per_image=8, aug_transforms=None):
        super(OurPatchLocalizationDataset, self).__init__(data_source, transform, samples_per_image)

        self.aug_transform = Compose(aug_transforms) if aug_transforms else Compose(RELIC_AUGMENTATIONS)

    def image_to_samples(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        img
            An image in torch.Tensor format.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            features, labels tuple
            features: torch.Tensor of shape (samples_per_image, 3, *img.size())
            labels: torch.Tensor of shape (samples_per_image) containing the corresponding labels (integers between 0 and 7)
        """
        # convert image into patches
        patches = image_to_patches(img)
        # randomly select n_samples from all possible labels without replacement
        labels = np.random.choice(8, self.samples_per_image)

        samples = []
        for label in labels:
            if label >= 4:
                # the middle patch (number 4) is never a neighbor
                label += 1
            samples.append(torch.stack((patches[4], self.aug_transform(patches[label]), self.aug_transform(patches[label]))))

        return torch.stack(samples), torch.from_numpy(labels)
