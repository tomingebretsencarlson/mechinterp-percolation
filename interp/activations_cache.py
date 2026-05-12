import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import argparse 
import yaml
import datetime
import shutil
import os, sys
from pathlib import Path
from datasets import build_datasets 
from models import build_model


def collect_activations(model, layer_name, dataloader, device, max_batches = None, flatten = True):

    model.eval()
    activations = []
    labels = []

    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Layer_name {layer_name} does not exist in the model. The availabel are: {list(modules)}")
    
    layer = modules[layer_name]

    def hook_fn(module, input, output):
        out = output.detach()
        if flatten and (out.ndim > 2):
            out = out.flatten(1)
        activations.append(out.cpu())

    hook_handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        counter = 0
        for x, y in dataloader:
            counter += 1
            x = x.to(device)
            _ = model(x)

            labels.append(y.cpu())

            if max_batches is not None and counter >= max_batches:
                break
    hook_handle.remove()
    return torch.cat(activations, dim = 0), torch.cat(labels, dim = 0)


def cache_activations_from_run(run_dir: Path, layer_name: str, device, splits = ("train","test"), max_batches=None, checkpoint=None):
    # Loading the config used
    with open(run_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Building and loading the model
    model = build_model(config).to(device)
    if checkpoint is not None:
        state = torch.load(run_dir / checkpoint, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    else:
        state = torch.load(run_dir / "model_final.pt", map_location=device)
        model.load_state_dict(state)
    model.eval()


    # Creating datasets
    all_data = {}

    for split in splits:
        train_set = (split == "train")
        dataset = build_datasets(config, trainset=train_set)  # note: trainset ignored for percolation, full dataset always returned
        batch_size = config["training"]["batch_size"]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = False)

        # Getting activations
        activations, labels = collect_activations(model, layer_name=layer_name, dataloader=dataloader, device=device , max_batches=max_batches)
        save_dir = run_dir / "activations"
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        ckpt_suffix = f"_{Path(checkpoint).stem}" if checkpoint is not None else ""
        torch.save(activations, save_dir / f"{split}_layer_{safe_name}{ckpt_suffix}_activations.pt")
        torch.save(labels, save_dir / f"{split}_layer_{safe_name}{ckpt_suffix}_labels.pt")

        all_data[split] = {
            "activations":activations,
            "labels":labels
        }
    return all_data
