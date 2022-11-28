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

from src.utils import create_logger, save_checkpoint, save_model


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
    device: str,
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
        train(model, train_loader, device, criterion, optimizer, epoch, logger, tb_writer, tb_dict, log_frequency)

        # evaluate on validation set
        acc = validate(model, val_loader, device, criterion, logger, tb_writer, tb_dict, log_frequency)

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
    device: str,
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

    # switch to training mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        curr_time = time.time()

        # reshape target shape from [batch_size, samples_per_image] to [batch_size * samples_per_image]
        target = target.view(-1).long().to(device) # cross entropy loss function expects long type

        # input has shape [batch_size, samples_per_image, n_patches, n_channels, img_height, img_width]
        if input.shape[2] == 2: # original pretext task
            center, neighbor = input[:,:,0,:,:,:], input[:,:,1,:,:,:]

            # reshape patch shapes from [batch_size, samples_per_image, n_channels, img_height, img_width]
            # to [batch_size * samples_per_image, n_channels, img_height, img_width]
            shape = center.shape
            center = center.view(-1, shape[2], shape[3], shape[4])
            neighbor = center.view(-1, shape[2], shape[3], shape[4])

            data_time.update(time.time() - curr_time) # record data loading time

            output = model(center.to(device), neighbor.to(device)) 
            loss = criterion(output, target) 
        else: # our pretext task
            center, neighbor1, neighbor2 = input[:,:,0,:,:,:], input[:,:,1,:,:,:], input[:,:,2,:,:,:]

            # reshape patch shapes from [batch_size, samples_per_image, n_channels, img_height, img_width]
            # to [batch_size * samples_per_image, n_channels, img_height, img_width]
            shape = center.shape
            center = center.view(-1, shape[2], shape[3], shape[4])
            neighbor1 = neighbor1.view(-1, shape[2], shape[3], shape[4])
            neighbor2 = neighbor2.view(-1, shape[2], shape[3], shape[4])

            data_time.update(time.time() - curr_time) # record data loading time

            output1, output2 = model(center.to(device), neighbor1.to(device), neighbor2.to(device))
            loss = criterion(output1, output2, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), center.size(0))

        # record batch processing time
        batch_time.update(time.time() - curr_time)

        # log after every `log_frequency` batches
        if i % log_frequency == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)*input.size(1)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)
            
            # update TensorBoard
            tb_writer.add_scalar('train_loss', losses.val, tb_dict['train_global_steps'])
            tb_dict['train_global_steps'] += 1


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    criterion: nn.Module,
    logger: logging.Logger,
    tb_writer: SummaryWriter,
    tb_dict: dict,
    log_frequency: int,
    ) -> float:
    """ Validate the model using the validation set and return the accuracy. """

    # keep track of batch processing time and losses
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluation mode
    model.eval()

    # for the computation of accuracy at the end
    all_preds = [] # all predicted class labels
    all_labels = [] # all true class labels

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            curr_time = time.time()

            # reshape target shape from [batch_size, samples_per_image] to [batch_size * samples_per_image]
            target = target.view(-1).long().to(device) # cross entropy loss function expects long type
            
            # input has shape [batch_size, samples_per_image, n_patches, n_channels, img_height, img_width]
            if input.shape[2] == 2: # original pretext task
                center, neighbor = input[:,:,0,:,:,:], input[:,:,1,:,:,:]

                # reshape patch shapes from [batch_size, samples_per_image, n_channels, img_height, img_width]
                # to [batch_size * samples_per_image, n_channels, img_height, img_width]
                shape = center.shape
                center = center.view(-1, shape[2], shape[3], shape[4])
                neighbor = center.view(-1, shape[2], shape[3], shape[4])

                output = model(center.to(device), neighbor.to(device)) 
                loss = criterion(output, target) 

                # update list of labels and predictions for computation of accuracy
                all_preds.append(torch.argmax(output, dim=1).cpu().numpy()) # class label = index of max logit
                all_labels.append(target.detach().cpu().numpy())
            else: # our pretext task
                center, neighbor1, neighbor2 = input[:,:,0,:,:,:], input[:,:,1,:,:,:], input[:,:,2,:,:,:]

                # reshape patch shapes from [batch_size, samples_per_image, n_channels, img_height, img_width]
                # to [batch_size * samples_per_image, n_channels, img_height, img_width]
                shape = center.shape
                center = center.view(-1, shape[2], shape[3], shape[4])
                neighbor1 = neighbor1.view(-1, shape[2], shape[3], shape[4])
                neighbor2 = neighbor2.view(-1, shape[2], shape[3], shape[4])

                output1, output2 = model(center.to(device), neighbor1.to(device), neighbor2.to(device))
                loss = criterion(output1, output2, target)

                # update list of labels and predictions for computation of accuracy
                all_preds.append(torch.argmax(output1, dim=1).cpu().numpy()) # class label = index of max logit
                all_preds.append(torch.argmax(output2, dim=1).cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())

            # record loss
            losses.update(loss.item(), center.size(0))

            # record batch processing time
            batch_time.update(time.time() - curr_time)

            # log after every `log_frequency` batches
            if i % log_frequency == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

        # Calculate accuracy for entire validation set
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        accuracy = (all_preds == all_labels).sum() / all_preds.shape[0]

        logger.info('Accuracy: {:.3f}'.format(accuracy))

        # update TensorBoard
        tb_writer.add_scalar('valid_loss', losses.avg, tb_dict['valid_global_steps'])
        tb_writer.add_scalar('valid_acc', accuracy, tb_dict['valid_global_steps'])
        tb_dict['valid_global_steps'] += 1

    return accuracy


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