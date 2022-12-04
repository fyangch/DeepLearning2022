# Enforcing Style Invariance in Patch Localization
### Team Members:
* Elior Ben Arous
* Dustin Brunner
* Jonathan Manz
* Felix Yang



## Overview:
This repository contains the code of our group project for the course *Deep Learning*. The notebooks in the root directory are meant for demonstrations and experiments. We put almost all of the main functionalities into the `src` directory such that the notebooks are as clean, minimal and comprehensible as possible.

### Credits:
We wrote almost all of the code ourselves from scratch. We thereby used [this repository by Microsoft](https://github.com/microsoft/human-pose-estimation.pytorch) as a guide for the structure of our training loop and for the logging of the training progress.


## Setup:
### ImageNet Dataset:
1. Create an account and log into the [ImageNet website](https://image-net.org/index.php).
2. Download the following [samples](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) and [labels](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz). Due to computational reasons, we use the ImageNet validation set as our dataset. Note: The labels are only used to get stratified subsets of the validation set with respect to the image classes such that we get a balanced dataset when experimenting with fewer samples.
3. Unzip the samples and labels and put both folders into the `data` directory. All images should be located at `./data/ILSVRC2012_img_val/*.JPEG`.

### TODO: CIFAR-10 or CIFAR-100 instructions
1. TODO

### Environment:
1. TODO


## Final Results:
To reproduce the final results from our report, please first follow the setup instructions above. Then, TODO




## Codebase Overview: 
### Notebooks:
* `PretextTaskTraining.ipynb`: This notebook is responsible for training our self-supervised models using the original patch-localization method as well as our own method.
* `StyleAugmentations.ipynb`: This experimental notebook visualizes the various transformations and augmentations we apply to the ImageNet images.

### Source Files:
* `src/dataset.py`: Our custom dataset classes.
* `src/loss.py`: Custom loss function for our proposed pretext tasks.
* `src/models.py`: Our models with support for different backbones.
* `src/train.py`: Training loop for both pretext and downstream tasks.
* `src/transforms.py`: Image transformations and augmentations.
* `src/utils.py`: Logging, model saving, checkpointing, etc.