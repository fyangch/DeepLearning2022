import optuna
from torchvision.transforms import Compose, RandomResizedCrop, RandomGrayscale, GaussianBlur, ColorJitter, RandomSolarize

from src.optuna import create_optuna_objective

aug_transform = Compose([
    RandomResizedCrop(size=224, scale=(0.32, 1.0), ratio=(0.75, 1.3333333333333333)),
    ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
    RandomGrayscale(p=0.05),
    GaussianBlur(kernel_size=23, sigma=(1e-10, 0.2)),
    RandomSolarize(0.7, p=0.2),
])

RUN_PRETEXT_PARAMS = {
    "experiment_id": "optuna_test",
    "pretext_type": "our",
    "aug_transform": aug_transform,
    "loss_alpha": 1,
    "loss_symmetric": True,
    "n_train": 46000,
    "optimizer_kwargs": {
        "lr": ("float-log", 1e-5, 1e-3),
        "weight_decay": 0,
    },
    "num_epochs": 50,
    "batch_size": 64,
    "num_workers": 4,
    "log_frequency": 100,
    "cache_images": True,
    "resume_from_checkpoint": False,
}

# maximal number of trials to perform
N_TRIALS = 1e9
# stops search if last trial ended more than TIMEOUT seconds after the start
TIMEOUT = 1e9

# create objective function
objective = create_optuna_objective(RUN_PRETEXT_PARAMS, save_models=False)

# create study
study = optuna.create_study(direction="maximize")

# run study
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)