from src.train import run_downstream
from src.utils import load_best_model
from src.models import OurPretextNetwork, OriginalPretextNetwork

# specify experiment id to load pretext model from
PRETEXT_EXPERIMENT_ID = "dustin_lr_1e4"

# load pretext model
pretext_model = load_best_model(PRETEXT_EXPERIMENT_ID, OurPretextNetwork(backbone="resnet18"))

optimizer_kwargs = {
    "lr": 1e-4,
    "weight_decay": 0,
}

run_downstream(
    experiment_id="HERE_YOUR_EXPERIMENT_ID",
    pretext_model=pretext_model,
    optimizer_kwargs=optimizer_kwargs,
    num_epochs=100,
    n_train=9000,
    resume_from_checkpoint=False,
    cache_images=True,
)
