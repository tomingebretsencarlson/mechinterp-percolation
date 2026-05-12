# Mechanistic Interpretability on Synthetic Percolation Data

This repo studies mechanistic interpretability in residual MLPs trained on
synthetic percolation-structured data. The goal is to understand what
computational structures emerge when a neural network learns to predict
a tree-weighted signal from high-dimensional spatial embeddings.

## Data

Using statistically self-similar synthetic datasets based on a percolation cluster model. Data is generated using [percolation-synthetic-data](https://github.com/aribrill/percolation-synthetic-data/tree/main).

## Structure
```
data/           – synthetic data generation and storage
datasets/       – PyTorch Dataset wrappers
models/         – MLP architectures (residual, transformer-style)
training/       – training loop, losses, optimizers, scheduler
interp/         – mechanistic interpretability tools
configs/        – YAML experiment configurations
  training/     – configs for NN training runs
  interp/       – configs for interpretability runs
analysis/       – notebooks and plotting utilities
scripts/        – entry points for training and evaluation
  train_nn.py                       – train a neural network
  train_sae.py                      – train a sparse autoencoder
  plot_training.py                  – generate training diagnostic plots
  filter_percolationdataset.py      – filter/subset datasets
  colab_training_nn_run_sweep.ipynb – Colab notebook for hand-picked sweeps
  colab_training_nn_run_rnd_search.ipynb – Colab notebook for random search
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

**Google Colab:**
```bash
pip install -r requirements_colab.txt
```

## Running Training

Training is fully configuration-driven via YAML files in `configs/training/`.

```bash
PYTHONPATH=. python scripts/train_nn.py --configpath configs/training/percolation.yaml
```

Outputs (model checkpoint, loss curves, config copy) are saved to `outputstraining/<run_name>_<timestamp>/`.

## Running on Google Colab

- `scripts/colab_training_nn_run_sweep.ipynb` — run hand-picked sweep configurations
- `scripts/colab_training_nn_run_rnd_search.ipynb` — run a random hyperparameter search

Both notebooks clone the repo, copy data from Google Drive, run training, and sync outputs back to Drive.

## Weights & Biases

Training supports optional W&B logging. Add your API key as a Colab secret named `WANDB_API_KEY`. Metrics logged include per-epoch train/val loss, grad norm, R², and the epoch at which val loss first beats the 1-NN baseline.

## Configuration

Key fields in a training config:

```yaml
seed: 42

output:
  name: my_run_name

dataset:
  name: percolation
  params:
    data_dir: data/percolation/gen_v2/
    data_file: percolation_dataset_size200000_dim100_seed0_1cluster.npz

model:
  name: transformer_style_mlp
  params:
    d_model: 100
    expansion: 4
    activation: gelu
    n_blocks: 3
    input_dim: 100
    output_dim: 1

training:
  loss: mse
  batch_size: 1024
  epochs: 2500
  max_patience: 2500
  compute_update_norm: false   # set true to track per-epoch parameter update norms (slower)
  scheduler:
    name: cosine               # cosine, reduce_on_plateau, or None
    T_max: 2500
    eta_min: 1e-7
  optimizer:
    name: adamw
    lr: 5e-5
    weight_decay: 0.01
```
