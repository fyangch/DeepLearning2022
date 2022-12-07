import torch
import torch.nn as nn
from src.dataset import OurPatchLocalizationDataset, OriginalPatchLocalizationDataset, sample_image_paths
from src.models import OriginalPretextNetwork, OurPretextNetwork
from src.loss import CustomLoss
from src.train import train_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: {}".format(device))
      
img_paths = sample_image_paths(frac=1.0)
ds_train= OurPatchLocalizationDataset(image_paths=img_paths[:46000], cache_images=True)
ds_val= OurPatchLocalizationDataset(image_paths=img_paths[46000:], cache_images=True)

print("Number of training images: \t {}".format(len(ds_train)))
print("Number of validation images: \t {}".format(len(ds_val)))

model= OurPretextNetwork(backbone="resnet18")
criterion= CustomLoss(alpha=1.0, symmetric=True)


train_model(
    experiment_id="our_pretext_elior_2",
    model=model,    
    ds_train=ds_train,
    ds_val=ds_val,
    device=device,
    criterion=criterion,
    optimizer=None,
    num_epochs=100,
    batch_size=64,
    num_workers=4,
    log_frequency=100,
    resume_from_checkpoint=False,
)
