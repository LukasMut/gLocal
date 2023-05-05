import torch
import numpy as np

Array = np.ndarray


def is_embedding_source(source):
    return source not in ["torchvision", "custom"]


def apply_transform(
    features: Array,
    transform: Array,
    things_mean: float,
    things_std: float,
    transform_type: str = None,
):
    features = (features - things_mean) / things_std
    features = features @ transform["weights"]
    if "bias" in transform:
        features += transform["bias"]
    features = torch.from_numpy(features)
    return features
