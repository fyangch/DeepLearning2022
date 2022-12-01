import os
import logging
import time
import csv
import torch
import torch.nn as nn
from torch.optim import Optimizer

from tensorboardX import SummaryWriter

from typing import Tuple


def create_logger_and_descr_file(experiment_id: str, experiment_descr: str) -> Tuple[logging.Logger, SummaryWriter]:
    # set up directory for the current experiment
    experiment_dir = os.path.join("out", experiment_id)
    log_dir = os.path.join(experiment_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # create text file with description of experiment
    file = open(os.path.join(experiment_dir, "description.txt"), "w")
    file.write(experiment_descr)
    file.close()

    # define filename for log file
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_fn = os.path.join(log_dir, f"{time_str}.log")
    
    # set up logger
    logging.basicConfig(filename=str(log_fn), format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # only add a stream handler if there isn't already one
    if len(logger.handlers) == 1: # <-- file handler is the existing handler
        console = logging.StreamHandler()
        logger.addHandler(console)

    # set up TensorBoard writer
    tb_writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    return logger, tb_writer


def save_plotting_data(experiment_id: str, metric: str, epoch: int, metric_val: float):
    """ Save metrics after each epoch in a CSV file (to create plots for our report later). """
    fn = os.path.join("out", experiment_id, f"{metric}.csv")

    # define header if file does not exist yet
    if not os.path.isfile(fn):
        with open(fn, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", metric])

    # append new data row
    with open(fn, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, metric_val])


# More details: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html 
def save_checkpoint(
    experiment_id: str, 
    next_epoch: int,
    best_acc: float,
    model: nn.Module,
    optimizer: Optimizer,
    filename: str="checkpoint.pth.tar"
    ):

    # checkpoint states
    d = {
        "next_epoch": next_epoch,
        "best_acc": best_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    experiment_dir = os.path.join("out", experiment_id)
    torch.save(d, os.path.join(experiment_dir, filename))


def save_model(model: nn.Module, experiment_id: str, filename: str):
    experiment_dir = os.path.join("out", experiment_id)
    torch.save(model.state_dict(), os.path.join(experiment_dir, filename))