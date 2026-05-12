import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class PercolationDataset(Dataset):
    def __init__(self, data_dir, data_file):

        project_root = Path(__file__).resolve().parents[1]

        data_path = project_root / data_dir / data_file
        res = np.load(Path(data_dir) / data_file)

        self.X = torch.tensor(res["X"], dtype=torch.float32)
        self.y = torch.tensor(res["y"], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
