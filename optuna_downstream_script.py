import optuna
from src.utils import load_best_model
from src.models import OurPretextNetwork
from src.optuna import create_optuna_objective

# specify experiment id to load pretext model from
PRETEXT_EXPERIMENT_ID = "dustin_lr_5e5"

# load pretext model
pretext_model = load_best_model(PRETEXT_EXPERIMENT_ID, OurPretextNetwork(backbone="resnet18"))

RUN_DOWNSTREAM_PARAMS = {
    "experiment_id": "optuna_downstream_test",
    "pretext_model": pretext_model,
    "use_aug_transform": ("int", 0, 1),
    "optimizer_kwargs": {
        "lr": ("float-log", 1e-5, 1e-3),
        "weight_decay": ("float-log", 1e-7, 1e-3),
    },
    "num_epochs": 30,
    "batch_size": ("int-log", 5, 8),
    "n_train": 9000,
    "cache_images": True,
    "num_workers": 4,
}

# maximal number of trials to perform
N_TRIALS = 1e9
# stops search if last trial ended more than TIMEOUT seconds after the start
TIMEOUT = 1e9

# create objective function
objective = create_optuna_objective(RUN_DOWNSTREAM_PARAMS)

# create study
study = optuna.create_study(direction="maximize")

# run study
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)