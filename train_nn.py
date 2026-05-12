import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import argparse 
import yaml
import datetime
import shutil
from pathlib import Path
from datasets import build_datasets 
from models import build_model
from training import build_loss, build_optimizer, build_scheduler, Trainer
from utils.data_split import split_dataset 
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Config input args.
parser = argparse.ArgumentParser()
parser.add_argument("--configpath", help="Set the path to the config file.")
args = parser.parse_args()

# Loading config
with open(args.configpath, 'r') as f:
    config = yaml.safe_load(f)
    print("Loaded config file.")

print("config: ", config)

# Init wandb
use_wandb = _WANDB_AVAILABLE and config.get("logging", {}).get("wandb", False)
if use_wandb:
    wandb.init(project="percolation", name=config["output"]["name"], config=config)


# Make run_dir and save config
base_name = config["output"]["name"]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path("outputstraining") / f"{base_name}_{timestamp}"
run_dir.mkdir(parents=True,exist_ok=False)
shutil.copy(args.configpath, run_dir / "config.yaml")
print(f"Made run_dir: {run_dir}")


# Set seed function - for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Setting seed
set_seed(config["seed"])

# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating datasets
dataset = build_datasets(config, trainset=True ) # note trainset parameter is ignored for percolation datasets.
train_dataset, val_dataset, _ = split_dataset(
    dataset,
    seed=config["seed"]
)

batch_size = config["training"]["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)

x_diag, y_diag = next(iter(val_loader))
x_diag, y_diag = x_diag.to(device), y_diag.to(device)

# Build model
model = build_model(config).to(device)
print("Model: ", model)

## Training ##

# build optimizer
optimizer = build_optimizer(config,model)

# build loss
loss_fn = build_loss(config)

# Verify initial loss
model.eval()
loss_total = 0.0
with torch.no_grad():
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss_total += loss_fn(output, y).item()

init_loss = loss_total / len(train_loader)
print(f"Initial loss: {init_loss:.6f} (expected ~1.02)")
if use_wandb:
    wandb.log({"init_loss": init_loss})

# build scheduler
scheduler = build_scheduler(config, optimizer)

# Training-loop
epochs = config["training"].get("epochs", 5)
max_patience = config["training"].get("max_patience", 5)
checkpoint_interval=config["training"].get("checkpoint_interval", 50)
max_grad_norm=config["training"].get("max_grad_norm", float('inf'))


trainer = Trainer(
    model = model,
    loss_fn = loss_fn,
    optimizer = optimizer,
    train_loader = train_loader,
    val_loader = val_loader,
    device = device,
    max_grad_norm = max_grad_norm,
    checkpoint_dir = run_dir,
    checkpoint_interval = checkpoint_interval,
    scheduler=scheduler,
    epochs = epochs,
    max_patience = max_patience,
    diag_batch=(x_diag, y_diag),
    diag_interval=config["training"].get("diag_interval", 10),
    compute_update_norm=config["training"].get("compute_update_norm", False),
)

train_losses, val_losses, total_grad_norms, total_update_norms = trainer.train()


# Save model
torch.save(model.state_dict(), run_dir / "model_final.pt")

# Save metrics
np.save(run_dir / "train_losses.npy", train_losses)
np.save(run_dir / "val_losses.npy", val_losses)
np.save(run_dir / "total_grad_norms.npy", total_grad_norms)
np.save(run_dir / "total_update_norms.npy", total_update_norms)

# Save diganostics
np.save(run_dir / "pred_snapshots.npy", trainer.pred_snapshots, allow_pickle=True)

if use_wandb:
    wandb.summary["final/val_loss"] = min(val_losses)
    wandb.summary["final/train_loss"] = min(train_losses)
    wandb.finish()

print(f"Outputs from training found in run_dir: {run_dir}")