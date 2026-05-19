import re
import numpy as np
import pandas as pd



def is_hidden_layer(name):
    return "mlpblock" in name


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
