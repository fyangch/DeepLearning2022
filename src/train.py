import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tensorboardX import SummaryWriter

import numpy as np
import random
import os

from typing import Optional

from utils import create_logger, save_checkpoint, save_model


# fix random seeds for reproducibility
# UNCOMMENT THIS BEFORE THE FINAL SUBMISSION!!!
"""seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)"""


# More TensorBoard details: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
def train_model(
    experiment_id: str, # e.g. "our_pretext_12"
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[Optimizer]=None,
    start_epoch: int=0, # only relevant if you resume from a checkpoint
    num_epochs: int=20,
    ) -> None:
    """ Training loop. """

    # Create text logger and TensorBoard writer
    logger, tb_writer = create_logger(experiment_id)
    tb_dict = { # dictionary for TensorBoard logging
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }

    # use Adam if no optimizer is specified
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    best_perf = 0.0 # tracks the best perforance so far
    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        train(model, train_loader, criterion, optimizer, epoch, tb_writer, tb_dict)

        # evaluate on validation set
        perf = validate() #TODO: insert args

        # save best model so far
        if perf > best_perf:
            best_perf = perf
            logger.info(f"Saving best model to ./out/{experiment_id}/")
            save_model(model, experiment_id, "best_model.pth.tar")

        # update checkpoint
        logger.info(f"Saving checkpoint to ./out/{experiment_id}/")
        save_checkpoint(experiment_id, epoch+1, model, optimizer)

    # save final model
    logger.info(f"Saving final model to ./out/{experiment_id}/")
    save_model(model, experiment_id, "final_model.pth.tar")

    # close TensorBoard writer
    tb_writer.close()
    

def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    tb_writer: SummaryWriter,
    tb_dict: dict,
    ) -> None:
    """ Train the model for one epoch. """
    # TODO
    return


def validate(
    # TODO
    ) -> float:
    """ Validate the model using the validation set and return some perfomance metric. """
    # TODO
    return