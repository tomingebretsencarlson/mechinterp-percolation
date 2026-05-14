import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import argparse
import re
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import is_hidden_layer

plt.rcParams.update({'font.size': 16})

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="Directory containing regresslat_depth_*.csv files")
parser.add_argument("--output_dir", default=None, help="Where to save plots. Defaults to input_dir.")
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir) if args.output_dir else input_dir
output_dir.mkdir(parents=True, exist_ok=True)

depth_files = sorted(input_dir.glob("regresslat_depth_*.csv"))
if not depth_files:
    raise FileNotFoundError(f"No regresslat_depth_*.csv files found in {input_dir}")

# Collect data: dict[layer_name] -> list of (depth, global_mse, global_r2, mean_baseline_mse)
layer_data = {}
depths_all = []


for f in depth_files:
    match = re.search(r"regresslat_depth_(\d+)\.csv", f.name)
    if not match:
        continue
    depth = int(match.group(1))
    depths_all.append(depth)
    df = pd.read_csv(f)
    for _, row in df.iterrows():
        layer = row["layer"]
        if layer not in layer_data:
            layer_data[layer] = []
        layer_data[layer].append({
            "depth": depth,
            "global_mse": row["global_mse"],
            "global_r2": row["global_r2"],
            "mean_baseline_mse": row["mean_baseline_mse"],
        })

depths_all = sorted(set(depths_all))

# Sort layers: put raw_x / input_layer first, then mlp_blocks in order
def layer_sort_key(name):
    if name == "raw_x":
        return (-1, -1)
    if name == "input_layer":
        return (0, 0)
    m = re.search(r"(\d+)", name)
    return (1, int(m.group(1))) if m else (2, 0)

layers_sorted = sorted(layer_data.keys(), key=layer_sort_key)

def block_color_key(name):
    if name in ("input_layer", "raw_x"):
        return 0
    m = re.search(r"mlp_blocks[._](\d+)", name)
    return int(m.group(1)) + 1 if m else 0

unique_color_keys = sorted(set(block_color_key(l) for l in layers_sorted))
mlp_keys = [k for k in unique_color_keys if k > 0]
color_map = {0: "black"}
for i, k in enumerate(mlp_keys):
    color_map[k] = plt.cm.viridis(0.1 + 0.75 * i / max(len(mlp_keys) - 1, 1))

color_handles = [
    mpatches.Patch(color=color_map[k], label="Input layer" if k == 0 else f"MLP block {k - 1}")
    for k in unique_color_keys
]
style_handles = [
    mlines.Line2D([], [], color="gray", marker="o", linestyle="-",  markersize=8, fillstyle="full", label="Residual stream"),
    mlines.Line2D([], [], color="gray", marker="s", linestyle=":", markersize=8, fillstyle="none", label="Hidden layer"),
]

def add_legend(ax):
    leg1 = ax.legend(handles=color_handles, fontsize=10, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, fontsize=10, loc="upper right")
# Ensure each layer has entries sorted by depth
for layer in layers_sorted:
    layer_data[layer].sort(key=lambda d: d["depth"])

# --- Plot 1: Global MSE vs depth ---
fig, ax = plt.subplots()
for i, layer in enumerate(layers_sorted):
    entries = layer_data[layer]
    xs = [e["depth"] for e in entries]
    ys = [e["global_mse"] for e in entries]
    ax.plot(xs, ys, marker="s" if is_hidden_layer(layer) else "o",
            linestyle=":" if is_hidden_layer(layer) else "-",
            color=color_map[block_color_key(layer)], fillstyle="none" if is_hidden_layer(layer) else "full")

# Baseline MSE (same for all layers at each depth — use first layer's values)
first_layer = layers_sorted[0]
baseline_xs = [e["depth"] for e in layer_data[first_layer]]
baseline_ys = [e["mean_baseline_mse"] for e in layer_data[first_layer]]
##ax.plot(baseline_xs, baseline_ys, color="red", linestyle="--", marker="x", label="Mean predictor baseline")

ax.set_xlabel("Latent depth")
ax.set_ylabel("MSE")
#ax.set_ylim([-0.005, 0.37])
add_legend(ax)
fig.tight_layout()
out_path = output_dir / "global_mse_by_depth.pdf"
fig.savefig(out_path, dpi=150)
out_path_png = output_dir / "global_mse_by_depth.png"
fig.savefig(out_path_png, dpi=150)
plt.close(fig)
print(f"Saved {out_path}")

# --- Plot 2: Global R² vs depth ---
fig, ax = plt.subplots()
for i, layer in enumerate(layers_sorted):
    entries = layer_data[layer]
    xs = [e["depth"] for e in entries]
    ys = [e["global_r2"] for e in entries]
    ax.plot(xs, ys, marker="s" if is_hidden_layer(layer) else "o",
            linestyle=":" if is_hidden_layer(layer) else "-",
            color=color_map[block_color_key(layer)], fillstyle="none" if is_hidden_layer(layer) else "full")

ax.set_xlabel("Latent depth")
ax.set_ylabel("$R^2$")
leg1 = ax.legend(handles=color_handles, fontsize=10, loc="lower left")
ax.add_artist(leg1)
ax.legend(handles=style_handles, fontsize=10, loc="lower right")
fig.tight_layout()
out_path = output_dir / "global_r2_by_depth.pdf"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved {out_path}")

# --- Plot 3: Incremental R² gain per layer vs depth ---
fig, ax = plt.subplots()
for i, layer in enumerate(layers_sorted):
    entries = layer_data[layer]
    xs = [e["depth"] for e in entries]
    r2s = [e["global_r2"] for e in entries]
    if i == 0:
        prev_r2s = [0.0] * len(r2s)
    else:
        prev_layer = layers_sorted[i - 1]
        prev_entries = {e["depth"]: e["global_r2"] for e in layer_data[prev_layer]}
        prev_r2s = [prev_entries.get(d, 0.0) for d in xs]
    deltas = [r2 - prev for r2, prev in zip(r2s, prev_r2s)]
    ax.plot(xs, deltas, marker="s" if is_hidden_layer(layer) else "o",
            linestyle=":" if is_hidden_layer(layer) else "-",
            color=color_map[block_color_key(layer)], fillstyle="none" if is_hidden_layer(layer) else "full")

ax.axhline(0.0, color="red", linestyle="--", label="No gain")
ax.set_xlabel("Latent depth")
ax.set_ylabel("$\\Delta R^2$ (vs previous layer)")
add_legend(ax)
fig.tight_layout()
out_path = output_dir / "delta_r2_by_depth.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved {out_path}")

print(f"\nAll plots saved to {output_dir}")
