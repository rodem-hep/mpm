# Masked particle modelling
[![python](https://img.shields.io/badge/-Python_3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/-PyTorch_2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra_1.3-89b8cd&logoColor=white)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/-WandB_0.16-orange?logo=weightsandbiases&logoColor=white)](https://wandb.ai)

This is the repository that was used for the Masked particle modelling paper [arxiv:2401.13537](https://arxiv.org/abs/2401.13537).

Though it has since deviated quite alot, this project was origianlly based on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

## Quickstart

 ```
# Clone project with vqtorch submodule
git clone --recurse-submodules https://gitlab.cern.ch/rodem/projects/mpm.git
cd mpm

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install requirements
pip install -r requirements.txt

# Install the vqtorch library
cd vqtorch
pip install .
```

This project requires python > 3.10.
The latest build is using PyTorch 2.0 and Lightning 2.0, but all needed python packages installed using the requirements.txt file. 
A docker image is also available on [docker hub](https://hub.docker.com/repository/docker/samklein/mpm_hep/general) with 

```docker pull samklein/mpm_hep:latest```.

The reccomended logger is WeightAndBiases `https://wandb.ai`, but there is the option to run with a csv logger.

# Running experiments
All experiments have an associated config file that can be found in the ```configs/experiments``` directory.

## Downloading data
The JetClass dataset is available here: https://zenodo.org/records/6619768

The RODEM data will be made publicly available soon.

## Setting paths
The path to where the JetClass data is stored must be specified by the ```jetclass_data``` key in the ```configs/paths/default.yaml``` config file.
In the same file you should specify the ```output_dir``` to which all results will be written.

## VQ-VAE Training

```
python scripts/train.py experiment=train_vq_vae.yaml project_name=mpm network_name=vq_vae
```

## MPM model training
The size of the pretrained model can be changed with the ```model.encoder``` key, and the type of pretraining task can be changed with the ```model``` key. Both of these point to configs in the ```configs/model``` directory.

```
python scripts/train.py experiment=train_mpm.yaml project_name=mpm network_name=pretrained paths.vq_vae_model=mpm/vq_vae 'model=vq_vae_mpm' model/encoder=backbone_small
```

## Fine tuning

```
python scripts/train.py experiment=train_fine.yaml paths.pretrained_model=mpm/pretrained model.train_backbone=True model.reinstantiate=False
```

If ```reinstantiate``` is set to ```True``` then the backbone weights will be resampled and all of the pretraining effect will be ignored.

If ```train_backbone``` is set to ```False``` then the backbone weights will be frozen during training.

### Project Structure
The directory structure of the project is as follows.

```
├── configs                  <- All hydra configs
│   ├── callbacks            <- Collection of lightning callbacks to run during training
│   ├── datamodule           <- Config for the lightning datamodule
│   ├── experiment           <- Single run experiment config
│   ├── hydra                <- Hydra config, can leave alone
│   ├── loggers              <- Collection of lightning loggers to run during training
│   ├── model                <- Model configuration
│   ├── paths                <- Project paths
│   ├── trainer              <- Lightning trainer class configuration
│   ├── export.yaml          <- Config for the export.py script
│   └── train.yaml           <- Config for the train.py script
├── docker          <- Docker build file
├── mattstools      <- mattstools folder with cross-project ML tools
├── README.md
├── requirements.txt
├── scripts                       <- All executable python scripts
│   ├── export_jetclass.py        <- Exports model outputs on the JetClass dataset
│   ├── export.py                 <- Exports a tagger based on configs/export.yaml
│   └── train.py                  <- Exports a tagger based on configs/train.yaml
└── src                           <- Main code for this project
```
