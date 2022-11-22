import os
import logging
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer

from tensorboardX import SummaryWriter

from typing import Tuple


def create_logger(experiment_id: str) -> Tuple[logging.Logger, SummaryWriter]:
    # set up directory for the current experiment
    experiment_dir = os.path.join("out", experiment_id)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # define filename for log file
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_fn = os.path.join(experiment_dir, f"{time_str}.log")
    
    # set up logger
    logging.basicConfig(filename=str(log_fn), format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # set up TensorBoard writer
    tb_writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    return logger, tb_writer


# More details: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html 
def save_checkpoint(
    experiment_id: str, 
    next_epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    filename: str="checkpoint.pth.tar"
    ):

    # checkpoint states
    d = {
        "next_epoch": next_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    experiment_dir = os.path.join("out", experiment_id)
    torch.save(d, os.path.join(experiment_dir, filename))


def save_model(model: nn.Module, experiment_id: str, filename: str):
    experiment_dir = os.path.join("out", experiment_id)
    torch.save(model.state_dict(), os.path.join(experiment_dir, filename))