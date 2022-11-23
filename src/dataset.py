import os

import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, RandomResizedCrop, RandomGrayscale, RandomHorizontalFlip, GaussianBlur, ColorJitter, RandomSolarize, ToPILImage, ToTensor, RandomCrop, CenterCrop, Resize
from tqdm import tqdm

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


def load_tiny_imagenet(tiny_imagenet_folder, dataset_type='train', transform=transforms.ToPILImage()):
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


def image_to_patches(img):
    """Crop split_per_side x split_per_side patches from input image.
    Args:
        img (Tensor image): input image.
    Returns:
        list[Tensor image]: A list of cropped patches.
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

    def __init__(self, data_source, transform=None, samples_per_image=8):
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

    def image_to_samples(self, img):
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

    def image_to_samples(self, img):
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
