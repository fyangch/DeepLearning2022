from src.train import run_downstream
from src.utils import load_best_model
from src.models import OriginalPretextNetwork, OurPretextNetwork, OurPretextNetworkv2

# specify experiment id to load pretext model from
PRETEXT_EXPERIMENT_ID = "PRETEXT_EXPERIMENT_ID"

# load pretext model
pretext_model = load_best_model(PRETEXT_EXPERIMENT_ID, OurPretextNetwork(backbone="resnet18"))

optimizer_kwargs = {
    "lr": 1e-4,
    "weight_decay": 0,
}

run_downstream(
    experiment_id="DOWNSTREAM_EXPERIMENT_ID",
    pretext_model=pretext_model,
    optimizer_kwargs=optimizer_kwargs,
    num_epochs=100,
    n_train=9000,
    resume_from_checkpoint=False,
    cache_images=True,
)
