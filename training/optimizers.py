import torch.optim as optim

def build_optimizer(config, model):

    opt_config = config["training"]["optimizer"].copy()
    name = opt_config["name"].lower()
    opt_config.pop("name")

    if name == "adam":
        return optim.Adam(model.parameters(), **opt_config)
    
    elif name == "adamw":
        return optim.AdamW(model.parameters(), **opt_config)
            
    
    elif name == "sgd":
        momentum = opt_config.get("momentum", 0,0)
        return optim.SGD(model.parameters(), **opt_config)
    
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    

def build_scheduler(config, optimizer):
    sched_config = config["training"].get("scheduler", None)

    if sched_config is None:
        return None
    name = (sched_config["name"] or "").lower()
    if name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=sched_config.get("patience", 5), factor=sched_config.get("factor", 0.5))
    elif name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sched_config.get("T_max", 1000), eta_min=sched_config.get("eta_min", 0.0) )
    return None
