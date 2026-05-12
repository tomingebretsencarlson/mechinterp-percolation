import torch

def split_dataset(dataset, seed, train_frac=0.8, val_frac=0.1):

    test_frac = 1 - train_frac - val_frac
    n = len(dataset)

    train_size = int(train_frac * n)
    val_size = int(val_frac * n)
    test_size = n - train_size - val_size

    g = torch.Generator().manual_seed(seed)

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=g
    )

    return train_ds, val_ds, test_ds