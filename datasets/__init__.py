"""
Helperfunction for datasets:
Builds datasets based on the configuration.
"""

def build_datasets(config, trainset = True):
    name = config["dataset"]["name"]
    params = config["dataset"]["params"]

    if name == "mnist":
        from .mnist import MNISTDataset
        return MNISTDataset(
            data_dir = params["data_dir"],
            trainset = trainset
        )
    
    elif config["dataset"]["name"] == "percolation":
        from .percolation import PercolationDataset
        return PercolationDataset(
            data_dir = params["data_dir"], 
            data_file = params["data_file"]
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {config["dataset"]["name"]}")
    