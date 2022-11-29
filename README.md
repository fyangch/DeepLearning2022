# Deep Learning Project
### Team Members:
* Elior Ben Arous
* Dustin Brunner
* Jonathan Manz
* Felix Yang



## Overview:
This repository contains the code of our group project for the course *Deep Learning*. The notebooks in the root directory are meant for demonstrations and experiments. We put almost all of the main functionalities into the `src` directory such that the notebooks are as clean, minimal and comprehensible as possible.

### Credits:
We wrote almost all of the code ourselves from scratch. We thereby used [this repository by Microsoft](https://github.com/microsoft/human-pose-estimation.pytorch) as a guide for the structure of our training loop, the normal logging, TensorBoard logging, as well as good PyTorch practices in general.



## Setup:
TODO: explain how to get dataset, set up environment --> requirements.txt, etc



## Final Predictions:
To reproduce the final results from our report, please first follow the setup instructions above. Then, TODO




## Codebase Overview: 
### Notebooks:
* `PretextTaskTraining.ipynb`: This notebook is responsible for training our self-supervised models using the original patch-localization method as well as our own method.
* `StyleAugmentations.ipynb`: This experimental notebook visualizes the various transformations and augmentations we apply to the ImageNet images.

### Source Files:
* `src/dataset.py`: Our custom dataset classes.
* `src/loss.py`: Custom loss function for our proposed pretext task.
* `src/models.py`: Our models with support for different backbones.
* `src/train_pretext.py`: Training loop for the pretext tasks.
* `src/transforms.py`: Image transformations and augmentations.
* `src/utils.py`: Logging, model saving, checkpointing.