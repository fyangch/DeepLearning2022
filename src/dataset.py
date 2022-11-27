import os

import cv2
import PIL
import skimage
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, RandomResizedCrop, RandomGrayscale, RandomHorizontalFlip, GaussianBlur, \
    ColorJitter, RandomSolarize, ToPILImage, RandomCrop, CenterCrop, Resize, ToTensor, Normalize

from typing import List, Union, Tuple

# recommended normalization parameters for ImageNet
IMAGENET_NORMALIZATION_PARAMS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

# tiny-imagenet-200 raw image transform
TINY_IMAGENET_RESIZE = Compose([
    ToTensor(),
    Resize(224),
])

# from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
# ImageNet raw image transform
IMAGENET_RESIZE = Compose([
    ToTensor(),
    Resize(256),
    CenterCrop(224),
])

IMAGENET_NORMALIZATION = Normalize(**IMAGENET_NORMALIZATION_PARAMS)

# random augmentations from ReLIC paper
RELIC_AUG_TRANSFORM = Compose([
    RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    RandomGrayscale(p=0.5),
    GaussianBlur(kernel_size=23, sigma=(0.1, 0.2)),
    RandomSolarize(0.5, p=0.5),
])


def load_tiny_imagenet(
        tiny_imagenet_folder: str,
        dataset_type: str = 'train',
        transform=ToPILImage()
) -> List[PIL.Image.Image]:
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
    List[PIL.Image.Image]
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


def sample_img_paths(
        labeldir: str = './data/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt',
        imagedir: str = './data/ILSVRC2012_img_val',
        frac: float = .1,
    ) -> pd.DataFrame:
    """
    Helper function to sample the ILSVRC2012_img_val dataset in a stratified method.
    Parameters
    ----------
    labeldir
        Path to the 'ILSVRC2012_validation_ground_truth.txt' file.
    imagedir
        Path to the 'ILSVRC2012_img_val' folder.
    frac
        Fraction of (image title, label) pairs that are kept in the sampling process compared to the initial entire dataset.
    Returns
    -------
    pd.DataFrame
        A DataFrame of the sampled Image titles and their corresponding label.
    """
    # Collect every class label for each image
    labels = pd.read_csv(labeldir, header=None).values.flatten()

    # Gather all image titles
    image_titles = os.listdir(imagedir)
    image_titles.sort()
    image_paths = [os.path.join(imagedir, image_title) for image_title in image_titles]

    # Create a Dataframe with the image titles and labels
    merge_dict = {'images': image_paths, 'labels': labels}
    df = pd.DataFrame(merge_dict)

    # Return a stratified sample of the dataset
    return df.groupby('labels', group_keys=False).apply(lambda x: x.sample(frac=frac, replace=False))


def image_to_patches(img: torch.Tensor) -> List[torch.Tensor]:
    """
    Cuts an image into 9 patches with 1/4 width and height of the original image and returns them in row major order.

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
            img_paths: List[str],
            pre_transform=None,
            samples_per_image: int = 8,
    ):
        """
        Parameters
        ----------
        img_paths
            A list of image paths returned by the sample_img_paths function.
        pre_transform
            A torchvision transform that will be applied to every raw image.
        samples_per_image
            How many samples to take from each image. Has to be an integer between 1 and 8.
        """
        self.img_paths = img_paths
        self.pre_transform = pre_transform if pre_transform else Compose([IMAGENET_RESIZE, IMAGENET_NORMALIZATION])
        self.samples_per_image = samples_per_image

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        # load image from path
        img_path = self.img_paths[idx]
        img = skimage.io.imread(img_path)
        # apply transform
        img = self.pre_transform(img)
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
        # convert image into 9 patches
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

    def __init__(
            self,
            img_paths: List[str],
            pre_transform=None,
            aug_transform=None,
            post_transform=None,
            samples_per_image: int = 8,
    ):
        """
        Parameters
        ----------
        img_paths
            A list of image paths returned by the sample_img_paths function.
        pre_transform
            A torchvision transform that is applied to every raw image BEFORE the augmentation.
        aug_transform
            Random style augmentation transform that is separately applied twice to the outer patch.
        pre_transform
            A torchvision transform that is applied to every augmented image AFTER the augmentation.
        samples_per_image
            How many samples to take from each image. Has to be an integer between 1 and 8.
        """
        super(OurPatchLocalizationDataset, self).__init__(img_paths=img_paths, pre_transform=pre_transform if pre_transform else IMAGENET_RESIZE, samples_per_image=samples_per_image)

        self.aug_transform = aug_transform if aug_transform else RELIC_AUG_TRANSFORM
        self.post_transform = post_transform if post_transform else IMAGENET_NORMALIZATION

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
            samples.append(
                torch.stack(
                    (self.post_transform(patches[4]),
                     self.post_transform(self.aug_transform(patches[label])),
                     self.post_transform(self.aug_transform(patches[label])))
                ))

        return torch.stack(samples), torch.from_numpy(labels)
