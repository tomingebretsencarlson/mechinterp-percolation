import torch.nn as nn


# Function to set the activation function. 
def get_activationfn(name):

    name = name.lower()

    if name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")