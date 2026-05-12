import torch.nn as nn 


def build_loss(config):

    loss_config = config["training"]["loss"].lower() 


    if loss_config == "mse":
        return nn.MSELoss()
    
    elif loss_config == "mae":
        return nn.L1Loss()
    
    elif loss_config == "cross_entropy":
        return nn.CrossEntropyLoss()
    
    else:
        raise ValueError(f"Unsupported loss: {loss_config}")