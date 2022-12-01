import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.io import ImageReadMode
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, RandomCrop, CenterCrop, Resize

from typing import List, Tuple

# project imports
from src.transforms import IMAGENET_RESIZE, RELIC_AUG_TRANSFORM, PATCH_LOCALIZATION_POST


def get_imagenet_info(
        data_dir: str = "data",
        savefile: str = os.path.join("data", "imagenet_info.csv"),
        recompute: bool = False,
    ) -> pd.DataFrame:
    """
    Helper function to get information (path, label, n_channels) about every image in the imagenet dataset.

    Parameters
    ----------
    data_dir
        Path to the 'data' directory. The 'data' directory should contain the following two directories:
        - 'ILSVRC2012_img_val': Containing the validation set images from the 2012 ILSVRC
        - 'ILSVRC2012_devkit_t12': Containing the developer kit from the 2012 ILSVRC
    savefile
        Path to where the imagenet_info pandas DataFrame should be loaded from and saved to.
    recompute
        Boolean indicating whether the imagenet_info should be recomputed even if it already exists.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the imagenet information.
    """

    # get image directory and path to labels
    image_dir = os.path.join(data_dir, "ILSVRC2012_img_val")
    label_path = os.path.join(data_dir, "ILSVRC2012_devkit_t12", "data", "ILSVRC2012_validation_ground_truth.txt")

    # check if imagenet_info already in data folder
    if not recompute and os.path.isfile(savefile):
        return pd.read_csv(savefile, index_col=0)

    # Collect every class label for each image
    labels = pd.read_csv(label_path, header=None).values.flatten()

    # Gather all image titles
    image_titles = os.listdir(image_dir)
    image_titles.sort()
    image_paths = [str(os.path.join(image_dir, image_title)) for image_title in image_titles]

    # Gather filter out non-RGB images (grayscale and RGBA)
    is_rgb = []
    for image_path in image_paths:
        img = torchvision.io.read_image(image_path)
        if len(img.shape) < 3 or img.shape[0] != 3:
            is_rgb.append(0)
        else:
            is_rgb.append(1)

    # Create a Dataframe with the image titles, labels and validity of image format
    merge_dict = {'images': image_paths, 'labels': labels, 'is_rgb': is_rgb}
    df = pd.DataFrame(merge_dict)

    # save imagenet info in data folder
    df.to_csv(savefile)

    return df


def sample_img_paths(
        frac: float = .1,
    ) -> np.ndarray:
    """
    Helper function to sample the ILSVRC2012_img_val dataset in a stratified method.
    Parameters
    ----------
    frac
        Fraction of (image title, label) pairs that are kept in the sampling process compared to the initial entire dataset.
    Returns
    -------
    np.ndarray
        A numpy array of the sampled image paths.
    """
    df = get_imagenet_info()

    # Only consider valid RGB images
    df = df[df['is_rgb'] == 1]

    # Return a stratified sample of the dataset
    return df.groupby('labels', group_keys=False).apply(lambda x: x.sample(frac=frac, replace=False))['images'].values


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

    # 1. center crop (ensure gap) 2. random crop (random jitter) 3. resize (to ensure img_size stays the same)
    random_jitter = Compose([CenterCrop(grid_size - patch_size // 4), RandomCrop(patch_size), Resize(img_size)])
    patches = [
        random_jitter(TF.crop(img, i * grid_size, j * grid_size, grid_size, grid_size))
        for i in range(splits_per_side)
        for j in range(splits_per_side)
    ]

    return patches


class OriginalPatchLocalizationDataset(Dataset):
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
        self.pre_transform = pre_transform if pre_transform else Compose([IMAGENET_RESIZE, PATCH_LOCALIZATION_POST])
        self.samples_per_image = samples_per_image

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        # load image from path
        img_path = self.img_paths[idx]
        img = torchvision.io.read_image(img_path, mode=ImageReadMode.RGB)/255
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
        self.post_transform = post_transform if post_transform else PATCH_LOCALIZATION_POST

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
