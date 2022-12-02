import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

import numpy as np
import time
import logging

from typing import Optional, List

from src.utils import fix_all_seeds, create_logger, save_plotting_data, save_checkpoint, load_checkpoint, save_model


def get_patches(batch_features: torch.Tensor, num_patches: int) -> List[torch.Tensor]:
    """ Select the individual patches from the features of a single batch and reshape them into the correct shape. """

    # batch_features have shape [batch_size, samples_per_image, n_patches, n_channels, img_height, img_width]
    patch_imgs = batch_features.view(-1, *batch_features.shape[-3:])

    # for each patch get every num_patches-th image
    return [patch_imgs[i::num_patches] for i in range(num_patches)]


def train_model(
    experiment_id: str, # e.g. "our_pretext_42"
    model: nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    device: str,
    criterion: nn.Module,
    optimizer: Optional[Optimizer]=None,
    num_epochs: int=100,
    batch_size: int=64,
    num_workers: int=4,
    resume_from_checkpoint: bool=False,
    log_frequency: int=10,
    fix_seed: bool=True, # training will not be reproducible if you resume from a checkpoint!
    seed: int=42,
    ) -> None:
    """ Training loop. """
    if fix_seed:
        fix_all_seeds(seed=seed)

    # create logger with file and console stream handlers
    logger = create_logger(experiment_id)

    # create data loaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # use Adam if no optimizer is specified
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    # resume from latest checkpoint if required
    start_epoch = 0
    best_acc = 0.0
    if resume_from_checkpoint:
        logger.info(f"Loading checkpoint from ./out/{experiment_id}/checkpoint.pth.tar")
        model, optimizer, start_epoch, best_acc = load_checkpoint(experiment_id, model, optimizer)
    
    # move model and criterion to GPU if available
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        train(experiment_id, model, train_loader, device, criterion, optimizer, epoch, logger, log_frequency)

        # evaluate on validation set
        acc = validate(experiment_id, model, val_loader, device, criterion, epoch, logger, log_frequency)

        # save best model so far
        if acc > best_acc:
            best_acc = acc
            logger.info(f"Saving best model to ./out/{experiment_id}/best_model.pth.tar")
            save_model(model, experiment_id, "best_model.pth.tar")

        # update checkpoint
        logger.info(f"Saving checkpoint to ./out/{experiment_id}/checkpoint.pth.tar")
        save_checkpoint(experiment_id, epoch+1, best_acc, model, optimizer)

    # save final model
    logger.info(f"Saving final model to ./out/{experiment_id}/final_model.pth.tar")
    save_model(model, experiment_id, "final_model.pth.tar")
    

def train(
    experiment_id: str,
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    criterion: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    logger: logging.Logger,
    log_frequency: int,
    ) -> None:
    """ Train the model for one epoch. """

    # keep track of batch processing time, data loading time and losses
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to training mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        curr_time = time.time()

        # reshape target shape from [batch_size, samples_per_image] to [batch_size * samples_per_image]
        target = target.view(-1).long().to(device) # cross entropy loss function expects long type
        input = input.to(device)

        # input has shape [batch_size, samples_per_image, n_patches, n_channels, img_height, img_width]
        if input.shape[2] == 2: # original pretext task
            center, neighbor = get_patches(input, 2)
            output = model(center, neighbor) 
            loss = criterion(output, target)
            # import matplotlib.pyplot as plt
            # print(f"label: {target.cpu()[0]}") 
            # plt.imshow(center[0].cpu().permute(1, 2, 0))
            # plt.show()
            # plt.imshow(neighbor[0].cpu().permute(1, 2, 0))
            # plt.show()
        elif input.shape[2] == 3: # our pretext task with 3 patches
            center, neighbor1, neighbor2 = get_patches(input, 3)
            output1, output2 = model(center, neighbor1, neighbor2)
            loss = criterion(output1, output2, target)
        else: # our pretext task with 4 patches
            center1, center2, neighbor1, neighbor2 = get_patches(input, 4)
            output1, output2 = model(center1, center2, neighbor1, neighbor2)
            loss = criterion(output1, output2, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss and batch processing time
        losses.update(loss.item(), center.size(0))
        batch_time.update(time.time() - curr_time)

        # log after every `log_frequency` batches
        if i % log_frequency == 0 or i == len(train_loader)-1:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader)-1, batch_time=batch_time,
                      speed=input.size(0)*input.size(1)/batch_time.val,
                      loss=losses)
            logger.info(msg)

    # save plotting data for later use
    save_plotting_data(experiment_id, "train_loss", epoch, losses.val)


def validate(
    experiment_id: str,
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    criterion: nn.Module,
    epoch: int,
    logger: logging.Logger,
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
            input = input.to(device)
            
            # input has shape [batch_size, samples_per_image, n_patches, n_channels, img_height, img_width]
            if input.shape[2] == 2: # original pretext task
                center, neighbor = get_patches(input, 2)
                output = model(center, neighbor) 
                loss = criterion(output, target) 

                # update list of labels and predictions for computation of accuracy
                all_preds.append(torch.argmax(output, dim=1).cpu().numpy()) # class label = index of max logit
                all_labels.append(target.detach().cpu().numpy())
            else: # our pretext tasks
                if input.shape[2] == 3: # our pretext task with 3 patches
                    center, neighbor1, neighbor2 = get_patches(input, 3)
                    output1, output2 = model(center, neighbor1, neighbor2)
                    loss = criterion(output1, output2, target)
                else: # our pretext task with 4 patches
                    center1, center2, neighbor1, neighbor2 = get_patches(input, 4)
                    output1, output2 = model(center1, center2, neighbor1, neighbor2)
                    loss = criterion(output1, output2, target)

                # update list of labels and predictions for computation of accuracy (our tasks contain 2 classifiaction tasks!)
                all_preds.append(torch.argmax(output1, dim=1).cpu().numpy()) # class label = index of max logit
                all_preds.append(torch.argmax(output2, dim=1).cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())
                all_labels.append(target.detach().cpu().numpy())

            # record loss and batch processing time
            losses.update(loss.item(), center.size(0))
            batch_time.update(time.time() - curr_time)

            # log after every `log_frequency` batches
            if i % log_frequency == 0 or i == len(val_loader)-1:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader)-1, batch_time=batch_time,
                          loss=losses)
                logger.info(msg)

        # Calculate accuracy for entire validation set
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        accuracy = (all_preds == all_labels).sum() / all_preds.shape[0]

        logger.info('Accuracy: {:.3f}'.format(accuracy))

        # save plotting data for later use
        save_plotting_data(experiment_id, "valid_loss", epoch, losses.val)
        save_plotting_data(experiment_id, "valid_acc", epoch, accuracy)

    return accuracy


# adopted from: https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/core/function.py
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