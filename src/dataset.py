import os

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
from torchvision.io import ImageReadMode
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, RandomCrop, CenterCrop, Resize

from typing import List, Tuple

# project imports
from src.transforms import IMAGENET_RESIZE, TINY_IMAGENET_TRANSFORM, RELIC_AUG_TRANSFORM, PATCH_LOCALIZATION_POST, RANDOM_JITTER_CROP, GRID_SIZE, \
    RelicAugmentationCreator


def get_imagenet_info(
        data_dir: str = "data",
        savefile: str = os.path.join("data", "imagenet_info.csv"),
        recompute: bool = False,
) -> pd.DataFrame:
    """
    Helper function to get information (path, label, n_channels) about every image in the imagenet validation dataset.

    Parameters
    ----------
    data_dir
        Path to the 'data' directory. The 'data' directory should contain the following two directories:
        - 'ILSVRC2012_img_val': Containing the validation set images from the 2012 ILSVRC (https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)
        - 'ILSVRC2012_devkit_t12': Containing the developer kit from the 2012 ILSVRC (https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz)
    savefile
        Path to where the imagenet_info pandas DataFrame should be loaded from and saved to.
    recompute
        Boolean indicating whether the imagenet_info should be recomputed even if it already exists.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the imagenet information.
    """

    # check if imagenet_info already in data folder
    if not recompute and os.path.isfile(savefile):
        # load df
        df = pd.read_csv(savefile, index_col=0)
        # Only consider valid RGB images
        df = df[df['is_rgb'] == 1]
        return df

    # get image directory and path to labels
    image_dir = os.path.join(data_dir, "ILSVRC2012_img_val")
    label_path = os.path.join(data_dir, "ILSVRC2012_devkit_t12", "data", "ILSVRC2012_validation_ground_truth.txt")

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

    # Only consider valid RGB images
    df = df[df['is_rgb'] == 1]

    return df


def get_tiny_imagenet_info(
        data_dir: str = "data",
        savefile: str = os.path.join("data", "tiny_imagenet_info.csv"),
        recompute: bool = False,
) -> pd.DataFrame:
    """
    Helper function to get information (path, label, n_channels) about every image in the tiny imagenet validation dataset.

    Parameters
    ----------
    data_dir
        Path to the 'data' directory. The 'data' directory should contain the following two directory:
        - 'tiny-imagenet-200': Containing the Tiny ImageNet dataset (https://image-net.org/data/tiny-imagenet-200.zip)
    savefile
        Path to where the tiny_imagenet_info pandas DataFrame should be loaded from and saved to.
    recompute
        Boolean indicating whether the tiny_imagenet_info should be recomputed even if it already exists.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the tiny imagenet information.
    """

    # check if imagenet_info already in data folder
    if not recompute and os.path.isfile(savefile):
        annotations_df = pd.read_csv(savefile, index_col=0)
        # Only consider valid RGB images
        annotations_df = annotations_df[annotations_df['is_rgb'] == 1]
        return annotations_df

    # get image directory and path to labels
    image_dir = os.path.join(data_dir, "tiny-imagenet-200", "val", "images")
    annotations_path = os.path.join(data_dir, "tiny-imagenet-200", "val", "val_annotations.txt")

    # load annotations file into pd dataframe
    annotations_df = pd.read_csv(annotations_path, sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
    classes = np.sort(annotations_df['Class'].unique())
    str_to_int_class = {s: idx for idx, s in enumerate(classes)}
    annotations_df['labels'] = annotations_df['Class'].apply(lambda x: str_to_int_class[x])

    # Gather all image titles
    image_paths = [str(os.path.join(image_dir, image_title)) for image_title in annotations_df['File']]

    # Gather filter out non-RGB images (grayscale and RGBA)
    is_rgb = []
    for image_path in image_paths:
        image = torchvision.io.read_image(image_path)
        if len(image.shape) < 3 or image.shape[0] != 3:
            is_rgb.append(0)
        else:
            is_rgb.append(1)

    # Create a Dataframe with the image titles, labels and validity of image format
    annotations_df['is_rgb'] = is_rgb
    annotations_df['images'] = image_paths

    # save imagenet info in data folder
    annotations_df.to_csv(savefile)

    # Only consider valid RGB images
    annotations_df = annotations_df[annotations_df['is_rgb'] == 1]

    return annotations_df


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
        random_jitter(F.crop(image, i * grid_size, j * grid_size, grid_size, grid_size))
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
    center_patch = RANDOM_JITTER_CROP(F.crop(image, 1 * GRID_SIZE, 1 * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    neighbor_patch = RANDOM_JITTER_CROP(
        F.crop(image, neighbor_row * GRID_SIZE, neighbor_col * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    return center_patch, neighbor_patch


class OriginalPatchLocalizationDataset(Dataset):
    """
    Dataset implementing the original Patch Localization method
    A sample is made up of the 8 possible tasks for a given grid ((center, neighbor), labels)
    """

    def __init__(
            self,
            imagenet_info: pd.DataFrame = None,
            pre_transform: nn.Module = None,
            post_transform: nn.Module = None,
            cache_images: bool = False,
    ):
        """
        Parameters
        ----------
        imagenet_info
            A pandas dataframe containing imagenet information returned by the get_imagenet_info function.
        pre_transform
            A torchvision transform that will be applied BEFORE caching the image.
        post_transform
            A torchvision transform that will be applied AFTER converting an image into a sample.
        cache_images
            Whether to cache the resized images after loading them for the first time or to reload them every time.
            Aims to reduce latency of reloading images at cost of more memory usage.
        """
        self.image_paths = imagenet_info['images'].values if imagenet_info is not None else get_imagenet_info()['images'].values
        self.pre_transform = pre_transform if pre_transform else IMAGENET_RESIZE
        self.post_transform = post_transform if post_transform else PATCH_LOCALIZATION_POST
        self.cache_images = cache_images
        self.image_cache = {}

        # if cache_images load all images into cache
        if self.cache_images:
            self.populate_cache()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):

        # load image
        image = self.load_image(idx)

        # randomly select a label (between 0 and 7)
        label = torch.randint(7, (1,)).item()
        # extract center patch and neighbor patch corresponding to label
        center_patch, neighbor_patch = extract_patches(image, label)

        # convert patches
        features = self.convert_patches(center_patch, neighbor_patch)

        # apply post transform to each patch
        features = [self.post_transform(patch) for patch in features]

        return features, label

    def populate_cache(self):
        # load all images into cache
        for idx, image_path in enumerate(self.image_paths):
            image = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB)
            # resize image
            image = self.pre_transform(image)
            self.image_cache[idx] = image

    def load_image(self, idx):
        # check whether caching is activated and idx is in cache
        if self.cache_images:
            # load from cache and convert to float
            image = self.image_cache[idx]
        else:
            # load image from path
            image_path = self.image_paths[idx]
            image = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB)
            # resize image
            image = self.pre_transform(image)

        # convert image to float
        image = image / 255

        return image

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
            imagenet_info: List[str],
            pre_transform: nn.Module = None,
            aug_transform: nn.Module = None,
            post_transform: nn.Module = None,
            cache_images: bool = False,
    ):
        """
        Parameters
        ----------
        imagenet_info
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
            imagenet_info=imagenet_info if imagenet_info is not None else get_imagenet_info(),
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


class OurPatchLocalizationDatasetv2(OriginalPatchLocalizationDataset):
    """
    Dataset implementing our modified Patch Localization method
    A sample is made up of the 8 possible tasks for a given grid ((center, A1(neighbor), A2(neighbor)), labels)
    """

    def __init__(
            self,
            imagenet_info: List[str],
            pre_transform: nn.Module = None,
            aug_transform_creator=None,
            post_transform: nn.Module = None,
            cache_images: bool = False,
    ):
        """
        Parameters
        ----------
        imagenet_info
            A list of image paths returned by the sample_image_paths function.
        pre_transform
            A torchvision transform that is applied to every raw image BEFORE the augmentation.
        aug_transform_creator
            An object with a method `get_random_function` that returns an augmentation transform for torch.Tensor.
        post_transform
            A torchvision transform that is applied to every augmented image AFTER the augmentation.
        cache_images
            Whether to cache the resized images after loading them for the first time or to reload them every time.
            Aims to reduce latency of reloading images at cost of more memory usage.
        """
        super(OurPatchLocalizationDatasetv2, self).__init__(
            imagenet_info=imagenet_info if imagenet_info is not None else get_imagenet_info(),
            pre_transform=pre_transform if pre_transform else IMAGENET_RESIZE,
            post_transform=post_transform if post_transform else PATCH_LOCALIZATION_POST,
            cache_images=cache_images,
        )

        self.aug_transform_creator = aug_transform_creator if aug_transform_creator else RelicAugmentationCreator()

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
            features: List with the 3 patches (A1(center), A1(neighbor), A2(center), A2(neighbor)) with shape [3, 224, 224]
        """
        # sample 2 random augmentations
        aug_transform1 = self.aug_transform_creator.get_random_function()
        aug_transform2 = self.aug_transform_creator.get_random_function()
        # augment the center and neighbor patch with both augmentations separately
        return [aug_transform1(center_patch), aug_transform1(neighbor_patch), aug_transform2(center_patch),
                aug_transform2(neighbor_patch)]


class DownstreamDataset(Dataset):
    """
    Dataset for the downstream image recognition task.
    """

    def __init__(
            self,
            tiny_imagenet_info: pd.DataFrame = None,
            transform: nn.Module = None,
            cache_images: bool = False,
    ):
        """
        Parameters
        ----------
        tiny_imagenet_info
            A pandas dataframe containing information about Tiny ImageNet returned by the get_tiny_imagenet_info function.
        transform
            A torchvision transform that will be applied to every image.
        cache_images
            Whether to cache the resized images after loading them for the first time or to reload them every time.
            Aims to reduce latency of reloading images at cost of more memory usage.
        """

        self.transform = transform if transform else TINY_IMAGENET_TRANSFORM
        self.cache_images = cache_images
        self.image_cache = {}

        tiny_imagenet_info = tiny_imagenet_info if tiny_imagenet_info is not None else get_tiny_imagenet_info()
        self.image_paths = tiny_imagenet_info['images'].values
        self.image_labels = tiny_imagenet_info['labels'].values

        # if cache_images load all images into cache
        if self.cache_images:
            self.populate_cache()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):

        # load image
        image = self.load_image(idx)

        # load label
        label = self.image_labels[idx]

        return image, label

    def populate_cache(self):
        # load all images into cache
        for idx, image_path in enumerate(self.image_paths):
            # store image
            self.image_cache[idx] = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB)

    def load_image(self, idx):
        # check whether caching is activated
        if self.cache_images:
            # load from cache and convert to float
            image = self.image_cache[idx] / 255
        else:
            # load image from path
            image_path = self.image_paths[idx]
            image = torchvision.io.read_image(image_path, mode=ImageReadMode.RGB) / 255
        # resize image
        image = self.transform(image)

        return image
