# Mechanistic Interpretability on Synthetic Percolation Data

This repo provides a framework for training neural networks on synthetic percolation-structured data and applying mechanistic interpretability tools to study their internal representations. The goal is to understand what computational structures emerge when a network learns to predict a hierarchically structured signal from high-dimensional embeddings.

Currently implemented: residual MLP training and layer-wise linear probing of activations against ground-truth latent variables.

## Data

Synthetic datasets based on a percolation cluster model with explicit ground-truth latent variables, designed as a testbed for mechanistic interpretability experiments.

Data is generated using [percolation-synthetic-data](https://github.com/aribrill/percolation-synthetic-data/tree/main).

## Structure
```
data/           – synthetic data storage
datasets/       – Dataset wrappers
models/         – model architectures (residual MLP, ...)
training/       – training loop, losses, optimizers, scheduler
interp/         – mechanistic interpretability tools
configs/        – YAML experiment configurations
  training/         – configs for model training runs
  interp/           – configs for interpretability runs
analysis/       – plotting scripts
scripts/        – entry points
  train_nn.py         – train a neural network
  train_probes.py     – train linear probes on layer activations
```

Output folders (not tracked in git, created on first run):
```
outputstraining/   – model checkpoints and loss curves
outputsinterp/     – interpretability results
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training a model

Training is fully configuration-driven via YAML files in `configs/training/`.

```bash
PYTHONPATH=. python scripts/train_nn.py --config_path configs/training/percolation.yaml
```

Outputs (model checkpoints, loss curves, config copy) are saved to `outputstraining/<run_name>_<timestamp>/`.

## Training linear probes

Probes are trained on cached layer activations and evaluated against ground-truth latent variables at each tree depth. Configure via YAML files in `configs/interp/`.

```bash
PYTHONPATH=. python scripts/train_probes.py --config_path configs/interp/probes.yaml
```

Outputs are saved to `outputsinterp/<run_name>_proberesults/<checkpoint_stem>/`.

## Configuration

Key fields in a training config:

```yaml
seed: 42

output:
  name: my_run

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
  max_patience: 5000   # set higher than epochs to disable early stopping
  diag_interval: 100   # log diagnostics every N epochs
  checkpoint_interval: 50
  compute_update_norm: true
  max_grad_norm: .inf  # gradient clipping; .inf disables it
  scheduler:
    name: cosine
    T_max: 500         # should match epochs for a full cosine decay
    eta_min: 0.000001
  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.01
```

Key fields in a probe config:

```yaml
seed: 42

output:
  name: my_probe_run

activations:
  input_run: outputstraining/my_run_<timestamp>
  checkpoint: checkpoint_epoch200.pt  # omit to use model_final.pt

probe:
  include_hidden: true   # also probe hidden states inside each block
  include_raw_x: true    # also probe raw input X
  params:
    depths: [10, 20, 40]  # tree depths to probe
    ridge_alpha: 1.0
    use_deltas: false           # probe block deltas (output - input) instead of absolute activations; only valid when include_hidden=false and include_raw_x=false
    min_latent_points: 50       # skip latents with fewer descendants / points
    partial_sum: false          # probe cumulative sum up to depth instead of individual latents
    features_path: data/percolation/genc_v1/percolation_dataset_size200000_dim100_seed0_gt_features.npz
```
