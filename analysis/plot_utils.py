import re
import numpy as np
import pandas as pd


def pretty_layer_name(name):
    if name == "input_layer":
        return "Input layer"
    if name == "raw_x":
        return "Raw input"
    m = re.match(r"mlp_blocks[._](\d+)[._]mlpblock[._]\d+", name)
    if m:
        return f"MLP block {m.group(1)} (hidden)"
    m = re.match(r"mlp_blocks[._](\d+)$", name)
    if m:
        return f"MLP block {m.group(1)}"
    return name


def is_hidden_layer(name):
    return "mlpblock" in name


def compute_median_bins(df_all, n_bins, energy_col="energy", score_col="score"):
    """
    Log-spaced binning of energy_col, returns median score_col per bin.
    Returns (bin_centers, medians, yerr, xerr, bins) or None if < 2 populated bins.
    yerr = [lower_err, upper_err] (IQR-based), xerr = [left_err, right_err] (bin half-widths).
    """
    df = df_all[df_all[energy_col] > 0].copy()
    if len(df) == 0:
        return None
    bins = np.logspace(np.log10(df[energy_col].min()), np.log10(df[energy_col].max()), n_bins + 1)
    df["_bin"] = pd.cut(df[energy_col], bins=bins, labels=False)
    bin_centers, medians, lower_err, upper_err, left_err, right_err = [], [], [], [], [], []
    for b in range(n_bins):
        sub = df[df["_bin"] == b][score_col]
        if len(sub) < 3:
            continue
        med = sub.median()
        center = np.sqrt(bins[b] * bins[b + 1])
        bin_centers.append(center)
        medians.append(med)
        lower_err.append(med - sub.quantile(0.25))
        upper_err.append(sub.quantile(0.75) - med)
        left_err.append(center - bins[b])
        right_err.append(bins[b + 1] - center)
    if len(bin_centers) < 2:
        return None
    return (np.array(bin_centers), np.array(medians),
            [np.array(lower_err), np.array(upper_err)],
            [np.array(left_err), np.array(right_err)], bins)


def pretty_layer_name_short(name):
    """Compact label — hidden vs residual stream explained in caption, not legend."""
    if name == "input_layer":
        return "Input layer"
    if name == "raw_x":
        return "Raw input"
    m = re.match(r"mlp_blocks[._](\d+)", name)
    if m:
        return f"MLP block {m.group(1)}"
    return name
