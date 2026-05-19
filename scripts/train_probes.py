import torch
import numpy as np
import random
from scipy import sparse
import argparse
import yaml
import shutil
import pandas as pd
from pathlib import Path
from interp.activations_cache import cache_activations_from_run
from interp.preprocessing import normalize, get_stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True, help="Give path to the probe training config.")
parser.add_argument("--shuffle_y", action="store_true", help="Shuffle targets as a sanity check. Probe should return MSE ≈ baseline and R² ≈ 0.")


args = parser.parse_args()


# Load config

with open(args.config_path, "r") as f:
    probe_config = yaml.safe_load(f)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(probe_config["seed"])

# Getting configs
model_run_dir = Path(probe_config["activations"]["input_run"])
print("Input run: ", model_run_dir)
max_batches = probe_config["activations"].get("max_batches", None)
checkpoint = probe_config["activations"].get("checkpoint", None)
print("Checkpoint: ", checkpoint if checkpoint is not None else "model_final.pt")
tree_heights = probe_config["probe"]["params"]["tree_heights"]
if isinstance(tree_heights, int):
    tree_heights = [tree_heights]
alpha = probe_config["probe"]["params"].get("ridge_alpha", 1.0)
use_deltas = probe_config["probe"]["params"].get("use_deltas", False)
min_latent_points = probe_config["probe"]["params"].get("min_latent_points", 1)
partial_sum = probe_config["probe"]["params"].get("partial_sum", False)
print("Training probe using deltas: ", use_deltas)
print(f"Probing tree heights: {tree_heights}")
print(f"Target mode: {'partial sum' if partial_sum else 'individual z_u'}")
print(f"Min points per latent: {min_latent_points}")

# Make run_dir and save probe_config
base_name = probe_config["output"]["name"]
checkpoint_stem = Path(checkpoint).stem if checkpoint is not None else "model_final"
parent_dir = Path("outputsinterp") / f"{base_name}_proberesults"
probe_run_dir = parent_dir / checkpoint_stem
probe_run_dir.mkdir(parents=True, exist_ok=False)
shutil.copy(args.config_path, probe_run_dir / "probe_config.yaml")
print(f"Made Probe run dir: {probe_run_dir}")

# Load model config and get layer names
with open(model_run_dir / "config.yaml", "r") as f:
    model_config = yaml.safe_load(f)

n_blocks = model_config["model"]["params"]["n_blocks"]
layer_names = [f"mlp_blocks.{i}" for i in range(n_blocks)]
if model_config["model"]["name"] in ("residual_mlp", "transformer_style_mlp"):
    layer_names = ["input_layer"] + layer_names
include_hidden = probe_config["probe"].get("include_hidden", False)
if include_hidden:
    interleaved = ["input_layer"] if "input_layer" in layer_names else []
    for i in range(n_blocks):
        interleaved.append(f"mlp_blocks.{i}.mlpblock.1")
        interleaved.append(f"mlp_blocks.{i}")
    layer_names = interleaved
include_raw_x = probe_config["probe"].get("include_raw_x", False)
if include_raw_x:
    layer_names = ["raw_x"] + layer_names
print(f"Sweeping layers: {layer_names}")

# Load raw input x if needed
dataset_params = model_config["dataset"]["params"]
dataset_path = Path(dataset_params["data_dir"]) / dataset_params["data_file"]
x_data = None
if include_raw_x:
    x_data = np.load(dataset_path)['X'].astype(np.float32)
    print(f"Loaded raw x: shape={x_data.shape}")
    
# Get features path from probe config, or derive from model config as fallback
features_path_cfg = probe_config["probe"]["params"].get("features_path", None)
if features_path_cfg is not None:
    features_path = Path(features_path_cfg)
else:
    params = model_config["dataset"]["params"]
    data_path = Path(params["data_dir"]) / params["data_file"]
    features_path = Path(str(data_path).replace("seed0", "seed0_gt_features"))
print(f"Features path: {features_path}")

# Load features with z_u values
features = sparse.load_npz(features_path)
print(f"Loaded features: {features.shape}")

# Load metadata for cluster_size
metadata_path = Path(str(features_path).replace("gt_features", "gt_metadata"))
meta_res = np.load(metadata_path, allow_pickle=True)
metadata_df = pd.DataFrame({k: meta_res[k] for k in meta_res.files})
metadata_cluster_size = metadata_df["cluster_size"].values
metadata_cluster_id = metadata_df["cluster_id"].values
print(f"Loaded metadata: {metadata_df.shape}")

# Activation cache dir
act_dir = model_run_dir / "activations"
act_dir.mkdir(parents=True, exist_ok=True)

first_layer = layer_names[0]
ckpt_suffix = f"_{Path(checkpoint).stem}" if checkpoint is not None else ""
if first_layer == "raw_x":
    first_acts = torch.tensor(x_data)
else:
    safe_name_first = first_layer.replace(".", "_").replace("/", "_")
    first_act_path = act_dir / f"train_layer_{safe_name_first}{ckpt_suffix}_activations.pt"
    if first_act_path.exists():
        first_acts = torch.load(first_act_path, map_location="cpu")
    else:
        all_acts = cache_activations_from_run(
            run_dir=model_run_dir,
            layer_name=first_layer,
            splits=("train",),
            device=device,
            max_batches=max_batches,
            checkpoint=checkpoint,
        )  # note: splits is ignored for percolation, full dataset always returned. Just set ("train",)
        first_acts = all_acts["train"]["activations"]

n_train = len(first_acts)

# Outer loop over depths (= tree_height)
for tree_height in tree_heights:
    print(f"\n{'='*50}")
    print(f"=== Tree height: {tree_height} ===")
    print(f"{'='*50}")

    # Build rows, z_u targets, and latent ids
    rows, targets, latent_ids = [], [], []
    for i in range(n_train):
        row = features.getrow(i)
        path_len = len(row.indices)
        in_window = path_len > tree_height 
        if in_window:
            rows.append(i)
            targets.append(float(row.data[:tree_height+1].sum()) if partial_sum else float(row.data[tree_height]))
            latent_ids.append(int(row.indices[tree_height]))  # which latent

    rows = np.array(rows)
    targets = np.array(targets, dtype=np.float32)
    latent_ids = np.array(latent_ids)

    # Filter to latents with at least min_latent_points
    unique_latents_raw, counts_raw = np.unique(latent_ids, return_counts=True)
    valid_latents = unique_latents_raw[counts_raw >= min_latent_points]
    valid_mask = np.isin(latent_ids, valid_latents)
    rows, targets, latent_ids = rows[valid_mask], targets[valid_mask], latent_ids[valid_mask]

    n_points = len(rows)
    unique_latents, counts = np.unique(latent_ids, return_counts=True)
    latent_to_cluster_size = {}
    latent_to_cluster_id = {}
    for i, lid in zip(rows, latent_ids):
        if lid not in latent_to_cluster_size:
            latent_to_cluster_size[lid] = int(metadata_cluster_size[i])
            latent_to_cluster_id[lid] = int(metadata_cluster_id[i])
    n_removed = len(unique_latents_raw) - len(unique_latents)
    print(f"Found {n_points} points with latent at depth {tree_height}, {len(unique_latents)} unique latents ({n_removed} removed with <{min_latent_points} pts)")
    print(f"Subtree sizes — min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")
    latent_to_value = {lid: targets[latent_ids == lid][0] for lid in unique_latents}
    latent_to_count = dict(zip(unique_latents, counts))

    if n_points == 0:
        print("No points remaining after filtering. Skipping.")
        continue

    # Shuffle to not bias training on cluster size
    rng = np.random.default_rng(probe_config["seed"])
    perm = rng.permutation(n_points)
    rows = rows[perm]
    targets = targets[perm]
    latent_ids = latent_ids[perm]

    # Sanity check for debugging
    if args.shuffle_y:
        print("[SANITY] Shuffling targets -> probe should return MSE approx baseline, R² approx 0")
        targets = rng.permutation(targets)

    val_size = int(0.3 * n_points)
    train_size = n_points - val_size
    y_train = targets[:train_size]
    y_val = targets[train_size:]
    latent_ids_val = latent_ids[train_size:]

    # Mean baseline
    mean_baseline_mse = mean_squared_error(y_val, np.full_like(y_val, y_train.mean()))
    print(f"Mean-prediction baseline MSE: {mean_baseline_mse:.6f}")

    # Sweep layers
    results = {}
    prev_activations = None

    for layer_name in layer_names:
        print(f"\n--- Layer: {layer_name} ---")
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        train_activations_path = act_dir / f"train_layer_{safe_name}{ckpt_suffix}_activations.pt"

        if layer_name == "raw_x":
            train_activations = torch.tensor(x_data)
            
        elif layer_name == first_layer:
            print("Reusing already loaded activations for first layer.")
            train_activations = first_acts
        elif train_activations_path.exists():
            print("Cached activations found. Loading...")
            train_activations = torch.load(train_activations_path, map_location="cpu")
        else:
            print("Collecting activations...")
            all_acts = cache_activations_from_run(
                run_dir=model_run_dir,
                layer_name=layer_name,
                splits=("train",),
                device=device,
                max_batches=max_batches,
                checkpoint=checkpoint,
            )# note: splits is ignored for percolation, full dataset always returned. Just set ("train",)
            train_activations = all_acts["train"]["activations"]

        assert train_activations.ndim == 2

        # Compute delta if use_deltas and not first layer
        if use_deltas and layer_name != first_layer:
            probe_tensor = train_activations - prev_activations
            print("  Using delta (output - input of block).")
        else:
            probe_tensor = train_activations

        # Normalize using train rows stats only
        probe_np = probe_tensor.detach().numpy()


        X_all = probe_np[rows]
        mean, std = get_stats(torch.tensor(X_all[:train_size]))
        probe_tensor = normalize(probe_tensor, mean, std)
        probe_np = probe_tensor.detach().numpy()

        X_all = probe_np[rows]
        X_train = X_all[:train_size]
        X_val = X_all[train_size:]


        reg = Ridge(alpha=alpha)
        reg.fit(X_train, y_train)
        preds_val = reg.predict(X_val)

        global_mse = mean_squared_error(y_val, preds_val)
        global_r2 = reg.score(X_val, y_val)

        # Per-latent MSE (with subtree size and mean-predictor baseline recorded)
        global_train_mean = y_train.mean()
        per_latent_rows = []
        for lid, count in zip(unique_latents, counts):
            mask = latent_ids_val == lid
            if mask.sum() > 0:
                mse = mean_squared_error(y_val[mask], preds_val[mask])
                z_u = latent_to_value[lid]
                baseline_mse = (global_train_mean - z_u) ** 2
                per_latent_rows.append({"latent_id": lid, "subtree_size": int(count), "cluster_size": latent_to_cluster_size.get(lid, 0), "cluster_id": latent_to_cluster_id.get(lid, 0), "mse": mse, "baseline_mse": baseline_mse})
  
        mean_per_latent_mse = np.mean([r["mse"] for r in per_latent_rows]) if per_latent_rows else float("nan")

        results[layer_name] = {
            "global_mse": global_mse,
            "global_r2": global_r2,
            "mean_per_latent_mse": mean_per_latent_mse,
            "per_latent": per_latent_rows,
        }
        print(f"  global_mse={global_mse:.6f}, r2={global_r2:.4f}, mean_per_latent_mse={mean_per_latent_mse:.6f}")

        prev_activations = train_activations
        del probe_tensor, probe_np, X_all, X_train, X_val

    # Save summary results
    rows_out = []
    for layer_name, metrics in results.items():
        rows_out.append({
            "layer": layer_name,
            "global_mse": metrics["global_mse"],
            "global_r2": metrics["global_r2"],
            "mean_per_latent_mse": metrics["mean_per_latent_mse"],
        })
    df = pd.DataFrame(rows_out)
    df["mean_baseline_mse"] = mean_baseline_mse
    df["n_points"] = n_points
    df["n_latents"] = len(unique_latents)
    df.to_csv(probe_run_dir / f"regresslat_depth_{tree_height}.csv", index=False)
    print(f"\nResults saved to {probe_run_dir / f'regresslat_depth_{tree_height}.csv'}")
    print(df[["layer", "global_mse", "global_r2", "mean_per_latent_mse"]].to_string(index=False))

    # Save per-latent MSE for all layers
    for layer_name, metrics in results.items():
        if metrics["per_latent"]:
            safe_name = layer_name.replace(".", "_").replace("/", "_")
            df_per_latent = pd.DataFrame(metrics["per_latent"])
            df_per_latent.to_csv(probe_run_dir / f"per_latent_mse_depth_{tree_height}_{safe_name}.csv", index=False)

print(f"\nProbe sweep done. Results in {probe_run_dir}")
