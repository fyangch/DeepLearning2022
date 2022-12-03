import torch
from torchvision.transforms import Compose, RandomResizedCrop, RandomGrayscale, RandomHorizontalFlip, GaussianBlur, \
    ColorJitter, RandomSolarize, CenterCrop, Resize, Normalize, RandomCrop
import torchvision.transforms.functional as F

from typing import Callable


class RandomColorDropping(torch.nn.Module):
    """
    Custom torchvision transform module

    Drop all except for one randomly chosen color channel from the image and replace the dropped channels
    with gaussian noise with 0 mean and `noise_std_factor` * torch.std(img[`channel_kept`]) standard deviation
    """

    def __init__(self, noise_std_factor: float = 0.01, inplace: bool = False):
        """
        Parameters
        ----------
        noise_std_factor
            The factor by which the standard deviation of the channel kept should be scaled to determine the
            standard deviation of the gaussian noise that the dropped channels are replaced with.
        inplace
            Whether to apply the transform directly to the input image or to a copy of it.
        """

        super().__init__()

        self.noise_std_factor = noise_std_factor
        self.inplace = inplace

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img
            An image in torch.Tensor format.

        Returns
        -------
        torch.Tensor
            Image with 2 randomly selected channels dropped.
        """

        assert img.shape[0] == 3, f"tensor inputs to RandomColorDropping must have 3 color channels, " \
                                  f"image passed has shape {img.shape}"

        if not self.inplace:
            img = torch.clone(img)

        # randomly determine a channel to be kept
        channel_kept = torch.randint(3, (1,)).item()
        channels_dropped = [i for i in range(3) if i != channel_kept]

        # replace dropped channels with gaussian noise
        img[channels_dropped] = torch.normal(mean=0, std=self.noise_std_factor * torch.std(img[channel_kept]),
                                             size=img[channels_dropped].shape)

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(channel_kept={self.channel_kept}, noise_std_factor={self.noise_std_factor})"


class ColorProjection(torch.nn.Module):
    """
    Custom torchvision transform module

    Project colors of an RGB image to avoid trivial solutions for the patch localization pretext task making use of the
    chromatic aberration in an image. Each pixel of the image is projected onto the green-magenta color axis as
    described on page 4 of the "Unsupervised Visual Representation Learning by Context Prediction" paper
    (https://arxiv.org/pdf/1505.05192.pdf).
    """

    def __init__(self):
        super().__init__()
        # define projection matrix
        a = torch.Tensor([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ])
        self.B = torch.eye(3) - a / 6

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img
            An RGB image in torch.Tensor format.

        Returns
        -------
        torch.Tensor
            Image with colors projected onto the green-magenta color axis.
        """
        return torch.einsum("ij,jhw->ihw", self.B, img)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# hardcode image input size 224x224
INPUT_SIZE = 224
GRID_SIZE = INPUT_SIZE // 3
PATCH_SIZE = INPUT_SIZE // 4


class RelicAugmentationCreator:
    """
    Class that constructs random ReLIC transformation functions.
    Using such a constructed function, the same randomness can be applied to multiple patches.
    """

    def __init__(self,
                 min_crop_scale: float = 0.8,
                 brightness: float = 0.1,
                 contrast: float = 0.1,
                 saturation: float = 0.1,
                 hue: float = 0.1,
                 grayscale_prob: float = 0.05,
                 kernel_size: int = 23,
                 sigma_max: float = 0.2,
                 solarize_prob: float = 0.2,
                 solarize_thresh: float = 0.5,
                 ):

        self.crop = RandomResizedCrop(size=244, scale=(min_crop_scale, 1.0))
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.grayscale_prob = grayscale_prob
        self.kernel_size = kernel_size
        self.sigma_max = sigma_max
        self.solarize_prob = solarize_prob
        self.solarize_thresh = solarize_thresh

        # transform random color jittering parameters to intervals
        self.brightness = ColorJitter._check_input(self, self.brightness, "brightness")
        self.contrast = ColorJitter._check_input(self, self.contrast, "contrast")
        self.saturation = ColorJitter._check_input(self, self.saturation, "saturation")
        self.hue = ColorJitter._check_input(self, self.hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def get_random_function(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """ Return a function that takes a tensor and returns an augmented tensor. """

        # fix the random color jittering parameters
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = ColorJitter.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        # determine whether to convert to grayscale
        to_grayscale = torch.rand(1).item() < self.grayscale_prob

        # fix the random blurring parameter
        sigma = GaussianBlur.get_params(1e-10, self.sigma_max)

        # determine whether to apply solarization
        apply_solarization = torch.rand(1).item() < self.solarize_prob

        # random augmentation function to return
        def func(img: torch.Tensor) -> torch.Tensor:
            # for the random resized crop it doesn't matter if the center and neighbor 
            # patches aren't cropped the EXACT same way
            img = self.crop(img)

            # color jittering (adopted from source code of the ColorJitter forward function)
            for fn_id in fn_idx:
                if fn_id == 0:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3:
                    img = F.adjust_hue(img, hue_factor)

            # random RGB to grayscale transformation
            if to_grayscale:
                img = F.rgb_to_grayscale(img, num_output_channels=3)

            # random Gaussian blurring
            img = F.gaussian_blur(img, self.kernel_size, sigma)

            # random solarization
            if apply_solarization:
                img = F.solarize(img, self.solarize_thresh)

            return img

        # return augmentation function with fixed random parameters
        return func


# tiny-imagenet-200 raw image transform
TINY_IMAGENET_RESIZE = Compose([
    Resize(224),
])

# from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
# ImageNet raw image transform
IMAGENET_RESIZE = Compose([
    Resize(256),
    CenterCrop(224),
])

# recommended normalization parameters for ImageNet
IMAGENET_NORMALIZATION_PARAMS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

# patch localization on imagenet post transform: 1. drop color channels (chromatic aberration) 2. normalize image
PATCH_LOCALIZATION_POST = Compose([
    ColorProjection(),
    Normalize(**IMAGENET_NORMALIZATION_PARAMS),
])

# random augmentations from ReLIC paper
RELIC_AUG_TRANSFORM = Compose([
    # RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    RandomGrayscale(p=0.05),
    GaussianBlur(kernel_size=23, sigma=(1e-10, 0.2)),
    # RandomSolarize(0.7, p=0.2),
])

# randomly crop a patch from a grid field with a PATCH_SIZE//4 gap
RANDOM_JITTER_CROP = Compose([
    CenterCrop(GRID_SIZE - PATCH_SIZE // 4),
    RandomCrop(PATCH_SIZE),
    Resize(INPUT_SIZE),
])
