# Code adopted from: https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/core/function.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tensorboardX import SummaryWriter

import numpy as np
import random
import os
import time
import logging

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
    log_frequency: int=10,
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

    best_acc = 0.0 # tracks the best accuracy so far
    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        train(model, train_loader, criterion, optimizer, epoch, logger, tb_writer, tb_dict, log_frequency)

        # evaluate on validation set
        acc = validate(model, val_loader, criterion, logger, tb_writer, tb_dict)

        # save best model so far
        if acc > best_acc:
            best_acc = acc
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
    logger: logging.Logger,
    tb_writer: SummaryWriter,
    tb_dict: dict,
    log_frequency: int,
    ) -> None:
    """ Train the model for one epoch. """

    # keep track of batch processing time, data loading time and losses
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        curr_time = time.time()

        # original pretext task
        if len(input) == 2: 
            center, neighbor = input[0], input[1]
            data_time.update(time.time() - curr_time) # measure data loading time

            output = model(center, neighbor) 
            loss = criterion(output, target) 

        # our pretext task
        else:
            center, neighbor1, neighbor2 = input[0], input[1], input[2]
            data_time.update(time.time() - curr_time) # measure data loading time

            output1, output2 = model(center, neighbor1, neighbor2)
            loss = criterion(output1, output2, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), center.size(0))

        # measure batch processing time
        batch_time.update(time.time() - curr_time)

        # log after every `log_frequency` batches
        if i % log_frequency == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)
            
            # update TensorBoard
            tb_writer.add_scalar('train_loss', losses.val, tb_dict['train_global_steps'])
            tb_dict['train_global_steps'] += 1


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    logger: logging.Logger,
    tb_writer: SummaryWriter,
    tb_dict: dict,
    ) -> float:
    """ Validate the model using the validation set and return the accuracy. """

    # switch to evaluate mode
    model.eval()


class AverageMeter(object):
    """ Computes and stores the average and current value. """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0