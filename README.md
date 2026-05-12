# Mechanistic Interpretability on Synthetic Percolation Data

This repo studies mechanistic interpretability in residual MLPs trained on
synthetic percolation-structured data. The goal is to understand what
computational structures emerge when a neural network learns to predict
a tree-weighted signal from high-dimensional spatial embeddings.

## Data

Using statistically self-similar synthetic datasets based on a percolation cluster model. Data is generated using [percolation-synthetic-data](https://github.com/aribrill/percolation-synthetic-data/tree/main).

## Structure
```
data/           – synthetic data storage
datasets/       – PyTorch Dataset wrappers
models/         – MLP architectures (residual mlp)
training/       – training loop, losses, optimizers, scheduler
interp/         – mechanistic interpretability tools
configs/        – YAML experiment configurations
  training/         – configs for NN training runs
  interp/           – configs for interpretability runs
analysis/       – notebooks and plotting utilities
scripts/        – entry points for training and evaluation
  train_nn.py         – train a neural network
  train_probes.py     - train probes
```

Output folders (not tracked in git, created on first run):
```
outputstraining/   – model checkpoints and loss curves
outputsinterp/     – interpretability results
```

## Setup

**Local:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training the NN 

Training is fully configuration-driven via YAML files in `configs/training/`.

```bash
PYTHONPATH=. python -m scripts.train_nn.py --config_path configs/training/percolation.yaml
```

Outputs (model checkpoint, loss curves, config copy) are saved to `outputstraining/<run_name>_<timestamp>/`.

## Training probes 


## Configuration

Key fields in a training config:

```yaml
seed: 42

output: 
  name: sweep_percolation_run_1_cluster_wd_0p01_lr1e-4consineetamin1e-6

dataset:
  name: percolation
  params:
      data_dir: data/percolation/genc_v1/
      data_file: percolation_dataset_size200000_dim100_seed0.npz

model:
  name: residual_mlp
  params:
    d_model: 256
    expansion: 4
    activation: relu
    n_blocks: 3
    input_dim: 100
    output_dim: 1

training:
  loss: mse
  batch_size: 1024
  epochs: 500
  max_patience: 3200 # Set higher than epochs to dissable 
  diag_interval: 100 # log diagnostics every N epochs
  checkpoint_interval: 50  # save model checkpoint every N epochs
  compute_update_norm: True  # log gradient update norms during training
  max_grad_norm: .inf # gradient clipping; .inf disables it
  scheduler:
    name: cosine
    patience: 50 # only used with ReduceLROnPlateau scheduler, ignored for cosine
    factor: 0.5 # only used with ReduceLROnPlateau scheduler, ignored for cosine
    T_max: 500  # should match epochs for a full cosine decay
    eta_min: 0.000001 # minimum learning rate at end of cosine schedule
  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.01
     



 
```
