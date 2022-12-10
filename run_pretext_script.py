from torchvision.transforms import Compose, RandomResizedCrop, ColorJitter, RandomGrayscale, GaussianBlur, RandomSolarize
from src.train import run_pretext

aug_transform = Compose([
    RandomResizedCrop(size=224, scale=(0.32, 1.0), ratio=(0.75, 1.3333333333333333)),
    ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
    RandomGrayscale(p=0.05),
    GaussianBlur(kernel_size=23, sigma=(1e-10, 0.2)),
    RandomSolarize(0.7, p=0.2),
])

optimizer_kwargs = {
    "lr": 1e-4,
    "weight_decay": 0,
}

run_pretext(
    experiment_id="HERE_YOUR_EXPERIMENT_ID",
    aug_transform=aug_transform,
    optimizer_kwargs=optimizer_kwargs,
    loss_alpha=1,
    loss_symmetric=True,
    pretext_type='our',
    cache_images=True,
    num_epochs=100,
    resume_from_checkpoint=False,
)
