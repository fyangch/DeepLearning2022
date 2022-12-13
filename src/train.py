import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer

import numpy as np
import time
import logging

from typing import Optional, Dict
from typing_extensions import Literal

from src.dataset import OurPatchLocalizationDataset, OurPatchLocalizationDatasetv2, OurPatchLocalizationDatasetv3, \
    OriginalPatchLocalizationDataset, get_imagenet_info, DownstreamDataset, get_tiny_imagenet_info
from src.loss import CustomLoss
from src.models import OurPretextNetwork, OurPretextNetworkv2, OriginalPretextNetwork, DownstreamNetwork
from src.utils import fix_all_seeds, create_logger, save_plotting_data, save_checkpoint, load_checkpoint, save_model


def train_model(
        experiment_id: str,
        model: nn.Module,
        ds_train: Dataset,
        ds_val: Dataset,
        device: str,
        criterion: nn.Module,
        optimizer: Optional[Optimizer] = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        num_workers: int = 4,
        resume_from_checkpoint: bool = False,
        log_frequency: int = 10,
        fix_seed: bool = True,  # training will not be reproducible if you resume from a checkpoint!
        seed: int = 42,
        logger: logging.Logger = None,
        save_models: bool = True,
) -> float:
    """ Training loop. """
    if fix_seed:
        fix_all_seeds(seed=seed)

    # create logger with file and console stream handlers
    if logger is None:
        logger = create_logger(experiment_id)

    # create data loaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # use Adam if no optimizer is specified
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    # move model and criterion to GPU if available
    model = model.to(device)
    criterion = criterion.to(device)

    # resume from latest checkpoint if required
    start_epoch = 0
    best_acc = 0.0
    if resume_from_checkpoint:
        logger.info(f"Loading checkpoint from ./out/{experiment_id}/checkpoint.pth.tar")
        model, optimizer, start_epoch, best_acc = load_checkpoint(experiment_id, model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        train(experiment_id, model, train_loader, device, criterion, optimizer, epoch, logger, log_frequency)

        # evaluate on validation set
        acc = validate(experiment_id, model, val_loader, device, criterion, epoch, logger, log_frequency)

        # save best model so far
        if acc > best_acc:
            best_acc = acc
            if save_models:
                logger.info(f"Saving best model to ./out/{experiment_id}/best_model.pth.tar")
                save_model(model, experiment_id, "best_model.pth.tar")

        if save_models:
            # update checkpoint
            logger.info(f"Saving checkpoint to ./out/{experiment_id}/checkpoint.pth.tar")
            save_checkpoint(experiment_id, epoch + 1, best_acc, model, optimizer)

    if save_models:
        # save final model
        logger.info(f"Saving final model to ./out/{experiment_id}/final_model.pth.tar")
        save_model(model, experiment_id, "final_model.pth.tar")

    # return best accuracy (for optuna)
    return best_acc


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

        target = target.long().to(device)  # cross entropy loss function expects long type

        if isinstance(input, list):  # pretext tasks
            if len(input) == 2:  # original pretext task
                center, neighbor = input
                output = model(center.to(device), neighbor.to(device))
                loss = criterion(output, target)
            elif len(input) == 3:  # our pretext task with 3 patches
                center, neighbor1, neighbor2 = input
                output1, output2 = model(center.to(device), neighbor1.to(device), neighbor2.to(device))
                loss = criterion(output1, output2, target)
            else:  # our pretext task with 4 patches
                center1, neighbor1, center2, neighbor2 = input
                output1, output2 = model(center1.to(device), neighbor1.to(device), center2.to(device),
                                         neighbor2.to(device))
                loss = criterion(output1, output2, target)
        else:  # downstream task
            output = model(input.to(device))
            loss = criterion(output, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss and batch processing time
        losses.update(loss.item(), target.size(0))
        batch_time.update(time.time() - curr_time)

        # log after every `log_frequency` batches
        if i % log_frequency == 0 or i == len(train_loader) - 1:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, i, len(train_loader) - 1, batch_time=batch_time,
                speed=target.size(0) / batch_time.val, loss=losses)
            logger.info(msg)

    # save plotting data for later use
    save_plotting_data(experiment_id, "train_loss", epoch, losses.avg)


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
    all_preds = []  # all predicted class labels
    all_labels = []  # all true class labels

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            curr_time = time.time()

            target = target.long().to(device)  # cross entropy loss function expects long type

            if isinstance(input, list):  # pretext tasks
                if len(input) == 2:  # original pretext task
                    center, neighbor = input
                    output = model(center.to(device), neighbor.to(device))
                    loss = criterion(output, target)

                    # update list of labels and predictions for computation of accuracy
                    all_preds.append(torch.argmax(output, dim=1).cpu().numpy())  # class label = index of max logit
                    all_labels.append(target.detach().cpu().numpy())
                else:  # our pretext tasks
                    if len(input) == 3:  # our pretext task with 3 patches
                        center, neighbor1, neighbor2 = input
                        output1, output2 = model(center.to(device), neighbor1.to(device), neighbor2.to(device))
                        loss = criterion(output1, output2, target)
                    else:  # our pretext task with 4 patches
                        center1, neighbor1, center2, neighbor2 = input
                        output1, output2 = model(center1.to(device), neighbor1.to(device), center2.to(device),
                                                 neighbor2.to(device))
                        loss = criterion(output1, output2, target)

                    # update list of labels and predictions for computation of accuracy (our tasks contain 2 classifiaction tasks!)
                    all_preds.append(torch.argmax(output1, dim=1).cpu().numpy())  # class label = index of max logit
                    all_preds.append(torch.argmax(output2, dim=1).cpu().numpy())
                    all_labels.append(target.detach().cpu().numpy())
                    all_labels.append(target.detach().cpu().numpy())
            else:  # downstream task
                output = model(input.to(device))
                loss = criterion(output, target)

                # update list of labels and predictions for computation of accuracy
                all_preds.append(torch.argmax(output, dim=1).cpu().numpy())  # class label = index of max logit
                all_labels.append(target.detach().cpu().numpy())

            # record loss and batch processing time
            losses.update(loss.item(), target.size(0))
            batch_time.update(time.time() - curr_time)

            # log after every `log_frequency` batches
            if i % log_frequency == 0 or i == len(val_loader) - 1:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    i, len(val_loader) - 1, batch_time=batch_time,
                    loss=losses)
                logger.info(msg)

        # Calculate accuracy for entire validation set
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        accuracy = (all_preds == all_labels).sum() / all_preds.shape[0]

        logger.info('Accuracy: {:.3f}'.format(accuracy))

        # save plotting data for later use
        save_plotting_data(experiment_id, "valid_loss", epoch, losses.avg)
        save_plotting_data(experiment_id, "valid_acc", epoch, accuracy)

    return accuracy


def run_pretext(
        experiment_id: str,
        aug_transform: nn.Module,
        pretext_type: Literal["our", "ourv2", "ourv3", "original"] = "our",
        loss_alpha: float = 1,
        loss_symmetric: bool = True,
        imagenet_info: pd.DataFrame = None,
        n_train: int = 46000,
        optimizer_kwargs: Dict = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        num_workers: int = 4,
        log_frequency: int = 100,
        cache_images: bool = True,
        resume_from_checkpoint: bool = False,
        save_models: bool = True,
) -> float:
    # initialize logger
    logger = create_logger(experiment_id)

    # print params used
    params = [item for item in locals().items() if item[0] not in ["imagenet_info"]]
    params_df = pd.DataFrame({"parameter": [p[0] for p in params], "value": [p[1] for p in params]})
    logger.info(params_df.to_markdown(index=False))

    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: {}".format(device))

    # if dataset not specified use whole dataset
    imagenet_info = imagenet_info if imagenet_info is not None else get_imagenet_info()

    # initialize datasets and model
    if pretext_type.lower() == "our":  # center, A1(neighbor), A2(neighbor)
        ds_train = OurPatchLocalizationDataset(imagenet_info=imagenet_info[:n_train], aug_transform=aug_transform,
                                               cache_images=cache_images)
        ds_val = OurPatchLocalizationDataset(imagenet_info=imagenet_info[n_train:], aug_transform=aug_transform,
                                             cache_images=cache_images)
        model = OurPretextNetwork(backbone="resnet18")
        criterion = CustomLoss(alpha=loss_alpha, symmetric=loss_symmetric)
    elif pretext_type.lower() == "ourv2":  # A1(center), A1(neighbor), A2(center), A2(neighbor)
        ds_train = OurPatchLocalizationDatasetv2(imagenet_info=imagenet_info[:n_train], cache_images=cache_images)
        ds_val = OurPatchLocalizationDatasetv2(imagenet_info=imagenet_info[n_train:], cache_images=cache_images)
        model = OurPretextNetworkv2(backbone="resnet18")
        criterion = CustomLoss(alpha=loss_alpha, symmetric=loss_symmetric)
    elif pretext_type.lower() == "ourv3":  # A1(center), A2(neighbor), A3(neighbor)
        ds_train = OurPatchLocalizationDatasetv3(imagenet_info=imagenet_info[:n_train], aug_transform=aug_transform,
                                                 cache_images=cache_images)
        ds_val = OurPatchLocalizationDatasetv3(imagenet_info=imagenet_info[n_train:], aug_transform=aug_transform,
                                               cache_images=cache_images)
        model = OurPretextNetwork(backbone="resnet18")  # for v3 we can use the same pretext model as for v1
        criterion = CustomLoss(alpha=loss_alpha, symmetric=loss_symmetric)
    else:  # original method
        ds_train = OriginalPatchLocalizationDataset(imagenet_info=imagenet_info[:n_train], cache_images=cache_images)
        ds_val = OriginalPatchLocalizationDataset(imagenet_info=imagenet_info[n_train:], cache_images=cache_images)
        model = OriginalPretextNetwork(backbone="resnet18")
        criterion = nn.CrossEntropyLoss()

    logger.info("Number of training images: \t {}".format(len(ds_train)))
    logger.info("Number of validation images: \t {}".format(len(ds_val)))

    # initialize optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    # train model
    best_acc = train_model(
        experiment_id=experiment_id,
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        log_frequency=log_frequency,
        resume_from_checkpoint=resume_from_checkpoint,
        logger=logger,
        save_models=save_models,
    )

    return best_acc


def run_downstream(
        experiment_id: str,
        pretext_model: OriginalPretextNetwork,
        tiny_imagenet_info: pd.DataFrame = None,
        use_aug_transform: bool = False,
        n_train: int = 9000,
        optimizer_kwargs: dict = None,
        num_epochs: int = 100,
        batch_size: int = 64,
        num_workers: int = 4,
        log_frequency: int = 100,
        cache_images: bool = True,
        resume_from_checkpoint: bool = False,
        save_models: bool = True,
) -> float:
    # initialize logger
    logger = create_logger(experiment_id)

    # print params used
    params = [item for item in locals().items() if item[0] not in ["pretext_model", "tiny_imagenet_info"]]
    params_df = pd.DataFrame({"parameter": [p[0] for p in params], "value": [p[1] for p in params]})
    logger.info(params_df.to_markdown(index=False))

    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: {}".format(device))

    # if dataset not specified use whole dataset
    tiny_imagenet_info = tiny_imagenet_info if tiny_imagenet_info is not None else get_tiny_imagenet_info()

    # initialize datasets
    ds_train = DownstreamDataset(tiny_imagenet_info=tiny_imagenet_info[:n_train], use_aug_transform=use_aug_transform,
                                 cache_images=cache_images)
    ds_val = DownstreamDataset(tiny_imagenet_info=tiny_imagenet_info[n_train:], use_aug_transform=use_aug_transform,
                               cache_images=cache_images)

    # initialize model
    model = DownstreamNetwork(pretext_model=pretext_model)

    logger.info("Number of training images: \t {}".format(len(ds_train)))
    logger.info("Number of validation images: \t {}".format(len(ds_val)))

    # initialize loss
    criterion = CrossEntropyLoss()

    # initialize optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    # train model
    best_acc = train_model(
        experiment_id=experiment_id,
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        log_frequency=log_frequency,
        resume_from_checkpoint=resume_from_checkpoint,
        logger=logger,
        save_models=save_models,
    )

    return best_acc


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
