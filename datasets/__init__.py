"""
Helperfunction for datasets:
Builds datasets based on the configuration.
"""

def build_datasets(config, trainset = True):
    params = config["dataset"]["params"]

    if config["dataset"]["name"] == "percolation":
        from .percolation import PercolationDataset
        return PercolationDataset(
            data_dir = params["data_dir"], 
            data_file = params["data_file"]
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {config["dataset"]["name"]}")
    