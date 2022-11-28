import torch
from torchvision.transforms import Compose, RandomResizedCrop, RandomGrayscale, RandomHorizontalFlip, GaussianBlur, \
    ColorJitter, RandomSolarize, CenterCrop, Resize, ToTensor, Normalize


class RandomColorDropping(torch.nn.Module):
    """
    Custom torchvision transform

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

# recommended normalization parameters for ImageNet
IMAGENET_NORMALIZATION_PARAMS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

# patch localization on imagenet post transform: 1. drop color channels (chromatic aberration) 2. normalize image
PATCH_LOCALIZATION_POST = Compose([
    RandomColorDropping(noise_std_factor=0.01, inplace=False),
    Normalize(**IMAGENET_NORMALIZATION_PARAMS)
])

# random augmentations from ReLIC paper
RELIC_AUG_TRANSFORM = Compose([
    RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    RandomGrayscale(p=0.5),
    GaussianBlur(kernel_size=23, sigma=(0.1, 0.2)),
    RandomSolarize(0.5, p=0.5),
])
