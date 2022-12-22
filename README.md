# Enforcing Style Invariance in Patch Localization
This repository contains the code of our group project for the course *Deep Learning (AS22 ETH Zürich)*.
### Team Members:
* Elior Ben Arous
* Dustin Brunner
* Jonathan Manz
* Felix Yang

### Credits:
We wrote most of the code ourselves from scratch. We thereby used [this repository by Microsoft](https://github.com/microsoft/human-pose-estimation.pytorch) as a guide for the structure of our training loop and for the logging of the training progress.


## Setup:
### ImageNet Dataset:
1. Create an account and log into the [ImageNet website](https://image-net.org/index.php).
2. Download the following [samples](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar) and [labels](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz). Due to computational reasons, we use the ImageNet validation set as our dataset. Note: The labels are only used to get stratified subsets of the validation set with respect to the image classes such that we get a balanced dataset when experimenting with fewer samples.
3. Unzip the samples and labels and put both folders into the `data` directory. All images should be located at `./data/ILSVRC2012_img_val/*.JPEG`.

### Tiny-ImageNet-200 Dataset:
1. Download and unzip the following [dataset](https://image-net.org/data/tiny-imagenet-200.zip).
2. Put the directory again into the `data` directory. It should be located at `./data/tiny-imagenet-200`.

### Environment:
1. Install miniconda ([installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).
2. Create new environment named `ESIPL` with python and activate it.
   1. `conda create -n ESIPL python`
   2. `conda activate ESIPL`
3. Install requirements listed in requirements.txt in the root directory of the repository.
   1. `pip install -r requirements.txt`
4. **Optional:** Install additional packages in the environment such as Jupyter Notebook or JupyterLab ([installation instructions](https://jupyter.org/install)).


## Final Results:
Follow these steps to train our final model and to reproduce our results from the downstream task:
1. Follow the setup instructions above.
2. Run `python run_pretext_script.py` to train our final pretext task model. All the best parameters are already set. Feel free to specify some pretext experiment ID (not necessary). Note: The training takes about 8.5 hours on the Euler cluster.
2. Run `python run_downstream_script.py` to evaluate the trained pretext task model on the downstream task. If changed the pretext experiment ID above, you need to use the same pretext ID in this script!


## Codebase Overview:
### Scripts:
* `run_pretext_script.py`: Script to run pretext task experiments.
* `run_downstream_script.py`: Script to run downstream task experiments.
* `optuna_pretext_script.py`: Script to find good hyperparameters for the pretext tasks using Optuna.
* `optuna_downstream_script.py`: Script to find good hyperparameters for the downstream tasks using Optuna.

### Notebooks:
* `StyleAugmentations.ipynb`: Visualizes the various transformations and augmentations we apply to the ImageNet images.
* `EmbeddingAnalysis.ipynb`: Analyses, compares and visualizes the embedding spaces of the original method and our method.
* `HyperparameterOptimization.ipynb`: Explains how to optimize hyperparameters using Optuna.

### Source Files:
* `src/dataset.py`: Our custom dataset classes for the pretext and downstream tasks.
* `src/loss.py`: Custom loss function for our proposed pretext tasks.
* `src/models.py`: Our pretext and downstream models with support for different backbones.
* `src/optuna.py`: Functions to find good hyperparameters with Optuna.
* `src/train.py`: Training loop for both pretext and downstream tasks.
* `src/transforms.py`: Image transformations and augmentations.
* `src/utils.py`: Logging, model saving, checkpointing, plotting, etc.