import os

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
from torchvision.io import ImageReadMode
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, RandomCrop, CenterCrop, Resize

from typing import List, Tuple

# project imports
from src.transforms import IMAGENET_RESIZE, RELIC_AUG_TRANSFORM, PATCH_LOCALIZATION_POST, RANDOM_JITTER_CROP, GRID_SIZE


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
        image = torchvision.io.read_image(image_path)
        if len(image.shape) < 3 or image.shape[0] != 3:
            is_rgb.append(0)
        else:
            is_rgb.append(1)

    # Create a Dataframe with the image titles, labels and validity of image format
    merge_dict = {'images': image_paths, 'labels': labels, 'is_rgb': is_rgb}
    df = pd.DataFrame(merge_dict)

    # save imagenet info in data folder
    df.to_csv(savefile)

    return df


def sample_image_paths(
        frac: float = .1,
) -> np.ndarray:
    """
    Helper function to sample the ILSVRC2012_image_val dataset in a stratified method.
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


def image_to_patches(image: torch.Tensor) -> List[torch.Tensor]:
    """
    Cuts an image into 9 patches with 1/4 width and height of the original image and returns them in row major order.

    Parameters
    ----------
    image
        torch.Tensor image to be split into patches.

    Returns
    -------
    List[torch.Tensor]
        A list of the 9 patches.
    """
    splits_per_side = 3  # split of patches per image side
    image_size = image.size()[-1]
    grid_size = image_size // splits_per_side
    patch_size = image_size // 4

    # 1. center crop (ensure gap) 2. random crop (random jitter) 3. resize (to ensure image_size stays the same)
    random_jitter = Compose([CenterCrop(grid_size - patch_size // 4), RandomCrop(patch_size), Resize(image_size)])
    patches = [
        random_jitter(TF.crop(image, i * grid_size, j * grid_size, grid_size, grid_size))
        for i in range(splits_per_side)
        for j in range(splits_per_side)
    ]

    return patches


def extract_patches(image: torch.Tensor, label: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crops the center and neighbor patch corresponding to a label between 0 and 7 (inclusive).

    Parameters
    ----------
    image
        torch.Tensor image with shape [3, 224, 224].
    label
        The label whose corresponding patch should be cropped. Has to be an integer between 0 and 7 (inclusive)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple of the center and the neighbor patch
    """
    assert 0 <= label <= 7, f"label has to be between 0 and 7 (inclusive), provided label: {label}"

    # calculate row and column of patch corresponding to label
    label = label if label < 4 else label + 1
    neighbor_row, neighbor_col = label // 3, label % 3

    # 1. crop center and neighbor patch with gaps and random jitter
    center_patch = RANDOM_JITTER_CROP(TF.crop(image, 1 * GRID_SIZE, 1 * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    neighbor_patch = RANDOM_JITTER_CROP(
        TF.crop(image, neighbor_row * GRID_SIZE, neighbor_col * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    return center_patch, neighbor_patch


class OriginalPatchLocalizationDataset(Dataset):
    """
    Dataset implementing the original Patch Localization method
    A sample is made up of the 8 possible tasks for a given grid ((center, neighbor), labels)
    """

    def __init__(
            self,
            image_paths: List[str],
            pre_transform: nn.Module = None,
            post_transform: nn.Module = None,
            cache_images: bool = False,
    ):
        """
        Parameters
        ----------
        image_paths
            A list of image paths returned by the sample_image_paths function.
        pre_transform
            A torchvision transform that will be applied BEFORE caching the image.
        post_transform
            A torchvision transform that will be applied AFTER converting an image into a sample.
        cache_images
            Whether to cache the resized images after loading them for the first time or to reload them every time.
            Aims to reduce latency of reloading images at cost of more memory usage.
        """
        self.image_paths = image_paths
        self.pre_transform = pre_transform if pre_transform else IMAGENET_RESIZE
        self.post_transform = post_transform if post_transform else PATCH_LOCALIZATION_POST
        self.cache_images = cache_images
        self.image_cache = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):

        # check whether caching is activated and idx is in cache
        if self.cache_images and idx in self.image_cache:
            # load from cache and convert to float
            image = self.image_cache[idx] / 255
        else:
            # load image from path
            image_path = self.image_paths[idx]
            image = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB)
            # resize image
            image = self.pre_transform(image)

            if self.cache_images:
                # if caching is activated, save uint8 image in cache
                self.image_cache[idx] = image

            # convert image to float
            image = image / 255

        # randomly select a label (between 0 and 7)
        label = torch.randint(7, (1,)).item()
        # extract center patch and neighbor patch corresponding to label
        center_patch, neighbor_patch = extract_patches(image, label)
        # convert patches
        features = self.convert_patches(center_patch, neighbor_patch)
        # apply post transform to each patch
        features = [self.post_transform(patch) for patch in features]

        return features, label

    def convert_patches(self, center_patch: torch.Tensor, neighbor_patch: torch.Tensor) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        center_patch
            A center patch in torch.Tensor format.
        neighbor_patch
            A neighbor patch in torch.Tensor format.

        Returns
        -------
        List[torch.Tensor]
            features: List with the 2 patches (center, neighbor) with shape [3, 224, 224]
        """

        return [center_patch, neighbor_patch]


class OurPatchLocalizationDataset(OriginalPatchLocalizationDataset):
    """
    Dataset implementing our modified Patch Localization method
    A sample is made up of the 8 possible tasks for a given grid ((center, A1(neighbor), A2(neighbor)), labels)
    """

    def __init__(
            self,
            image_paths: List[str],
            pre_transform: nn.Module = None,
            aug_transform: nn.Module = None,
            post_transform: nn.Module = None,
            cache_images: bool = False,
    ):
        """
        Parameters
        ----------
        image_paths
            A list of image paths returned by the sample_image_paths function.
        pre_transform
            A torchvision transform that is applied to every raw image BEFORE the augmentation.
        aug_transform
            Random style augmentation transform that is separately applied twice to the outer patch.
        post_transform
            A torchvision transform that is applied to every augmented image AFTER the augmentation.
        cache_images
            Whether to cache the resized images after loading them for the first time or to reload them every time.
            Aims to reduce latency of reloading images at cost of more memory usage.
        """
        super(OurPatchLocalizationDataset, self).__init__(
            image_paths=image_paths,
            pre_transform=pre_transform if pre_transform else IMAGENET_RESIZE,
            post_transform=post_transform if post_transform else PATCH_LOCALIZATION_POST,
            cache_images=cache_images,
        )

        self.aug_transform = aug_transform if aug_transform else RELIC_AUG_TRANSFORM

    def convert_patches(self, center_patch: torch.Tensor, neighbor_patch: torch.Tensor) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        center_patch
            A center patch in torch.Tensor format.
        neighbor_patch
            A neighbor patch in torch.Tensor format.

        Returns
        -------
        List[torch.Tensor]
            features: List with the 3 patches (center, A1(neighbor), A2(neighbor)) with shape [3, 224, 224]
        """

        return [center_patch, self.aug_transform(neighbor_patch), self.aug_transform(neighbor_patch)]
