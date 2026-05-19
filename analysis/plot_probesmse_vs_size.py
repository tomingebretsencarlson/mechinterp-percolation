import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import argparse
import re
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import pretty_layer_name_short, is_hidden_layer

plt.rcParams.update({'font.size': 16})

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="Directory containing per_latent_mse_depth_*.csv files")
parser.add_argument("--output_dir", default=None, help="Where to save plots. Defaults to input_dir.")
parser.add_argument("--min_baseline_mse", type=float, default=1e-4, help="Filter latents with baseline_mse below this threshold.")
parser.add_argument("--min_subtree_size", type=int, default=0, help="Filter latents with subtree_size below this threshold.")
parser.add_argument("--x_min_fit", type=float, default=0.0, help="Filter points below x_min_fit below this threshold for the powerlaw fit.")
parser.add_argument("--n_bins", type=int, default=5, help="Number bins in the medain MSE plot.")
parser.add_argument("--debug", action = "store_true", help="Adds debug print outs.")

args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir) if args.output_dir else input_dir
output_dir.mkdir(parents=True, exist_ok=True)

# Collect all per-latent CSVs: per_latent_mse_depth_{depth}_{layer}.csv
per_latent_files = sorted(input_dir.glob("per_latent_mse_depth_*.csv"))
if not per_latent_files:
    raise FileNotFoundError(f"No per_latent_mse_depth_*.csv files found in {input_dir}")


# Parse depth and layer, collect into dict[layer]: list of DFs
layer_records = {}
for f in per_latent_files:
    match = re.search(r"per_latent_mse_depth_(\d+)_(.+)\.csv", f.name)
    if not match:
        continue
    depth = int(match.group(1))
    layer = match.group(2)
    df = pd.read_csv(f)
    df["depth"] = depth
    df["energy"] = df["subtree_size"] / np.sqrt(df["cluster_size"])
    df["score"] = df["mse"]
    if layer not in layer_records:
        layer_records[layer] = []
    layer_records[layer].append(df)



# Sort layers
def layer_sort_key(name):
    if name == "raw_x":
        return (-1, -1)
    if name == "input_layer":
        return (0, 0)
    m = re.search(r"(\d+)", name)
    return (1, int(m.group(1))) if m else (2, 0)

layers_sorted = sorted(layer_records.keys(), key=layer_sort_key)
print("Sorted layers: ", layers_sorted)


# Print and plot median MSE per energy bin per layer
print("\n=== Median MSE per energy bin per layer ===")
n_bins = args.n_bins

def block_color_key(name):
    if name in ("input_layer", "raw_x"):
        return 0
    m = re.search(r"mlp_blocks_(\d+)", name)
    return int(m.group(1)) + 1 if m else 0

unique_color_keys = sorted(set(block_color_key(l) for l in layers_sorted))
mlp_keys = [k for k in unique_color_keys if k > 0]
color_map = {0: "black"}
for i, k in enumerate(mlp_keys):
    color_map[k] = plt.cm.viridis(0.1 + 0.75 * i / max(len(mlp_keys) - 1, 1))

has_raw_x = "raw_x" in layer_records
if has_raw_x:
    fig_med, (ax_med, ax_ratio) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
else:
    fig_med, ax_med = plt.subplots(figsize=(8, 5))
    ax_ratio = None

# Plotting median mse per energy bin
for li, layer in enumerate(layers_sorted):
    if layer == "raw_x":
        continue
    dfs = layer_records[layer]
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[all_df["baseline_mse"] >= args.min_baseline_mse]
    all_df = all_df[all_df["subtree_size"] >= args.min_subtree_size]
    all_df_e = all_df[all_df["energy"] > 0].copy()
    bins = np.logspace(np.log10(all_df_e["energy"].min()), np.log10(all_df_e["energy"].max()), n_bins + 1)
    all_df_e["energy_bin"] = pd.cut(all_df_e["energy"], bins=bins, labels=False)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    medians, yerr_lower, yerr_upper = [], [], []
    print(f"\n{layer}:")
    for b in range(n_bins):
        sub = all_df_e[all_df_e["energy_bin"] == b]["score"]
        med = sub.median()
        q25, q75 = sub.quantile(0.25), sub.quantile(0.75)
        medians.append(med)
        yerr_lower.append(med - q25)
        yerr_upper.append(q75 - med)
        print(f"  bin {b} [{bins[b]:.2e}, {bins[b+1]:.2e}): median={med:.4f}, IQR=[{q25:.4f}, {q75:.4f}], n={len(sub)}")
    medians = np.array(medians)
    yerr = [np.array(yerr_lower), np.array(yerr_upper)]
    xerr = [bin_centers - bins[:-1], bins[1:] - bin_centers]
    marker = "s" if is_hidden_layer(layer) else "o"
    fillstyle = "none" if is_hidden_layer(layer) else "full"
    color = color_map[block_color_key(layer)]
    ax_med.errorbar(bin_centers, medians, yerr=yerr, xerr=xerr, fmt=marker, linestyle="none",
                    color=color, fillstyle=fillstyle, markersize=8,
                    label=pretty_layer_name_short(layer), capsize=0)

    # Powerlaw fit
    if layer == "mlp_blocks_2" and not any(np.isnan(medians)) and (medians > 0).all():
        from scipy import stats
        fit_mask = bin_centers >= args.x_min_fit
        slope, intercept, r, _, _ = stats.linregress(np.log10(bin_centers[fit_mask]), np.log10(medians[fit_mask]))
        x_fit = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 50)
        y_fit = 10 ** (intercept + slope * np.log10(x_fit))
        fit_line, = ax_med.plot(x_fit, y_fit, ":", color="black", linewidth=2.5, label="_nolegend_", zorder=10)
        fit_label = f"MLP block 2 fit: slope={slope:.2f}"
        print(f"\nPower law fit for {layer}: slope={slope:.3f}, R²={r**2:.3f}")
        print("Medians: ", medians)
        print("yerr: ", yerr)

ax_med.set_xscale("log")
ax_med.set_yscale("log")
ax_med.set_xlim(0.3, 480)
if not has_raw_x:
    ax_med.set_xlabel(r"$s_\mathrm{latent} / \sqrt{s_\mathrm{cluster}}$")
ax_med.set_ylabel("MSE")
color_handles = []
for k in unique_color_keys:
    label = "Input layer" if k == 0 else f"MLP block {k - 1}"
    color_handles.append(mpatches.Patch(color=color_map[k], label=label))
if 'fit_line' in dir():
    fit_line.set_label(fit_label)
    color_handles.append(fit_line)
leg1 = ax_med.legend(handles=color_handles, fontsize=10, loc="upper right")
ax_med.add_artist(leg1)

style_handles = [
    mlines.Line2D([], [], color="gray", marker="o", linestyle="none", markersize=8, fillstyle="full", label="Residual stream"),
    mlines.Line2D([], [], color="gray", marker="s", linestyle="none", markersize=8, fillstyle="none", label="Hidden layer"),
]
ax_med.legend(handles=style_handles, fontsize=10, loc="lower left")
if not has_raw_x:
    out_path = output_dir / f"median_mse_by_energy_mss{args.min_subtree_size}_xm{args.x_min_fit}.pdf"
    out_path_png = output_dir / f"median_mse_by_energy_mss{args.min_subtree_size}_xm{args.x_min_fit}.png"
    fig_med.tight_layout()
    fig_med.savefig(out_path, dpi=150)
    fig_med.savefig(out_path_png, dpi=150)
    plt.close(fig_med)
    print(f"\nSaved {out_path}")

# Ratio panel (normalized by raw_x median per bin) 
if has_raw_x:
    raw_x_df = pd.concat(layer_records["raw_x"], ignore_index=True)
    raw_x_df = raw_x_df[raw_x_df["baseline_mse"] >= args.min_baseline_mse]
    raw_x_df = raw_x_df[raw_x_df["subtree_size"] >= args.min_subtree_size]
    raw_x_df = raw_x_df[raw_x_df["energy"] > 0].copy()
    shared_bins = np.logspace(np.log10(raw_x_df["energy"].min()), np.log10(raw_x_df["energy"].max()), n_bins + 1)
    shared_bin_centers = np.sqrt(shared_bins[:-1] * shared_bins[1:])
    raw_x_lookup = raw_x_df.set_index(["latent_id", "depth"])["score"]

    for layer in layers_sorted:
        if layer == "raw_x":
            continue
        dfs = layer_records[layer]
        all_df = pd.concat(dfs, ignore_index=True)
        all_df = all_df[all_df["baseline_mse"] >= args.min_baseline_mse]
        all_df = all_df[all_df["subtree_size"] >= args.min_subtree_size]
        all_df_e = all_df[all_df["energy"] > 0].copy()
        all_df_e = all_df_e.join(raw_x_lookup.rename("raw_x_score"), on=["latent_id", "depth"])
        all_df_e["ratio"] = all_df_e["score"] / all_df_e["raw_x_score"]
        if args.debug:
            print("\n")
            print("layer: ", layer)
            print("raw_x_lookup:\n ", raw_x_lookup)
            print("all_df_e[[latent_id, score, raw_x_score, ratio]]: \n", all_df_e[["latent_id", "score", "raw_x_score", "ratio"]])

        all_df_e["energy_bin"] = pd.cut(all_df_e["energy"], bins=shared_bins, labels=False)
        medians_norm, yerr_lower, yerr_upper = [], [], []
        for b in range(n_bins):
            sub = all_df_e[all_df_e["energy_bin"] == b]["ratio"]
            med = sub.median()
            medians_norm.append(med)
            yerr_lower.append(med - sub.quantile(0.25))
            yerr_upper.append(sub.quantile(0.75) - med)

        medians_norm = np.array(medians_norm)
   
        xerr = [shared_bin_centers - shared_bins[:-1], shared_bins[1:] - shared_bin_centers]
        marker = "s" if is_hidden_layer(layer) else "o"
        fillstyle = "none" if is_hidden_layer(layer) else "full"
        color = color_map[block_color_key(layer)]
        ax_ratio.errorbar(shared_bin_centers, medians_norm,
                        yerr=[np.array(yerr_lower), np.array(yerr_upper)], xerr=xerr,
                        fmt=marker, linestyle="none", color=color, fillstyle=fillstyle, markersize=8, capsize=0)
  
    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax_ratio.set_ylim(0.0, 1.5)
    ax_ratio.set_xlabel(r"$s_\mathrm{latent} / \sqrt{s_\mathrm{cluster}}$")
    ax_ratio.set_ylabel(r"MSE / MSE$_\mathrm{raw\ x}$", fontsize=12)

    out_path = output_dir / f"median_perlatent_mse_by_mss{args.min_subtree_size}_xm{args.x_min_fit}.pdf"
    out_path_png = output_dir / f"median_perlatent_mse_mss{args.min_subtree_size}_xm{args.x_min_fit}.png"
    fig_med.tight_layout()
    fig_med.savefig(out_path, dpi=150)
    fig_med.savefig(out_path_png, dpi=150)
    plt.close(fig_med)
    print(f"\nSaved {out_path}")
else:
    print("No raw_x data: saved standalone median MSE plot without ratio panel")

