from torchvision.transforms import Compose, RandomResizedCrop, ColorJitter, RandomGrayscale, GaussianBlur, RandomSolarize
from src.train import run_pretext

aug_transform = Compose([
    RandomResizedCrop(size=224, scale=(0.32, 1.0), ratio=(0.75, 1.3333333333333333)),
    ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
    RandomGrayscale(p=0.05),
    GaussianBlur(kernel_size=23, sigma=(1e-10, 0.2)),
    RandomSolarize(0.5, p=0.2),
])

optimizer_kwargs = {
    "lr": 5e-5,
    "weight_decay": 0,
}

run_pretext(
    experiment_id="PRETEXT_EXPERIMENT_ID",
    aug_transform=aug_transform,
    optimizer_kwargs=optimizer_kwargs,
    loss_alpha=5,
    loss_symmetric=True,
    pretext_type='our',
    cache_images=False,
    num_epochs=100,
    resume_from_checkpoint=False,
)
