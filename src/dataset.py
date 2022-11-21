import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, RandomResizedCrop, RandomGrayscale, RandomHorizontalFlip, GaussianBlur, RandomSolarize, ColorJitter


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


def image_to_patches(img):
    """Crop split_per_side x split_per_side patches from input image.
    Args:
        img (PIL Image): input image.
    Returns:
        list[PIL Image]: A list of cropped patches.
    """
    splits_per_side = 3  # split of patches per image side
    h, w = img.size()[1:]
    h_grid = h // splits_per_side
    w_grid = w // splits_per_side

    patches = [
        TF.crop(img, i * h_grid, j * w_grid, h_grid, w_grid)
        for i in range(splits_per_side)
        for j in range(splits_per_side)
    ]

    return patches


class OriginalPatchLocalizationDataset(torch.utils.data.Dataset):
    """
    Dataset implementing the original Patch Localization method
    A sample is made up of the 8 possible tasks for a given grid ((center, neighbor), labels)
    """

    def __init__(self, data_source, transform=None):
        self.data_source = data_source
        self.transform = transform if transform else TINY_IMAGENET_TRANSFORM

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        img = self.data_source[idx]

        img = self.transform(img)
        # split image into 9 patches
        patches = image_to_patches(img)
        # generate the permutations between the patches
        perms = self.generate_perms(patches)
        # labels for patch pairs
        patch_labels = torch.LongTensor(list(range(8)))

        samples = (torch.stack(perms), patch_labels)
        return samples

    def generate_perms(self, patches):
        return [torch.cat((patches[4], patches[i]), dim=0) for i in range(9) if i != 4]


class OurPatchLocalizationDataset(OriginalPatchLocalizationDataset):
    """
    Dataset implementing our modified Patch Localization method
    A sample is made up of the 8 possible tasks for a given grid ((center, random_aug(neighbor), random_aug(neighbor)), labels)
    """

    def __init__(self, data_source, transform=None, aug_transforms=None):
        super(OurPatchLocalizationDataset, self).__init__(data_source, transform)

        self.aug_transform = Compose(aug_transforms) if aug_transforms else Compose(RELIC_AUGMENTATIONS)

    def generate_perms(self, patches):
        # randomly transform outer patch in two different ways
        return [
            torch.cat((TF.resize(patches[4], 224), self.aug_transform(patches[i]), self.aug_transform(patches[i])), dim=0)
            for i in range(9) if i != 4
        ]
