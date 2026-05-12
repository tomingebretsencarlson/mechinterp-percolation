import torch


def get_stats(x):
    # Calculating mean and std for each activation across all activations 
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0,keepdim=True) + 1e-6
    return mean, std


def normalize(x, mean, std):
    x_norm = (x - mean)/std
    return x_norm

def denormalize(x_norm, mean, std):
    return x_norm*std + mean
