#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import pickle
import re
from collections import Counter
from functools import partial
from typing import Tuple

import flax
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import serialization
from ml_collections import config_dict

from data.data_partitioner import DataPartitioner


Array = jnp.ndarray
FrozenDict = config_dict.FrozenConfigDict


def load_data(root: str, file: str) -> dict:
    with h5py.File(os.path.join(root, file), "r") as f:
        data = {k: f[k][:] for k in f.keys()}
    return data


def uniform(n_classes):
    return np.ones(n_classes) * (1 / n_classes)


def load_metrics(metric_path):
    """Load pretrained parameters into memory."""
    binary = find_binaries(metric_path)
    metrics = pickle.loads(open(os.path.join(metric_path, binary), "rb").read())
    return metrics


def save_params(out_path, params, epoch):
    """Encode parameters of network as bytes and save as binary file."""
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    bytes_output = serialization.to_bytes(params)
    with open(os.path.join(out_path, f"pretrained_params_{epoch}.pkl"), "wb") as f:
        pickle.dump(bytes_output, f)


def save_opt_state(out_path, opt_state, epoch):
    """Encode parameters of network as bytes and save as binary file."""
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    bytes_output = serialization.to_bytes(opt_state)
    with open(os.path.join(out_path, f"opt_state_{epoch}.pkl"), "wb") as f:
        pickle.dump(bytes_output, f)


def find_binaries(param_path):
    """Search for last checkpoint."""
    param_binaries = sorted(
        [
            f
            for _, _, files in os.walk(param_path)
            for f in files
            if re.search(r"(?=.*\d+)(?=.*pkl$)", f)
        ]
    )
    return param_binaries.pop()


def get_epoch(binary):
    return int("".join(c for c in binary if c.isdigit()))


def merge_params(pretrained_params, current_params):
    return flax.core.FrozenDict(
        {"encoder": pretrained_params["encoder"], "clf": current_params["clf"]}
    )


def get_val_set(dataset, data_path) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if dataset == "cifar10":
        data = np.load(os.path.join(data_path, "validation.npz"))
        X = data["data"]
        y = data["labels"]
    else:
        data = torch.load(os.path.join(data_path, "validation.pt"))
        X = data[0].numpy()
        y = data[1].numpy()
    X = jnp.array(X)
    X = X.reshape(X.shape[0], -1)
    y = jax.nn.one_hot(y, jnp.max(y) + 1)
    return (X, y)


def get_full_dataset(partitioner: object) -> Tuple[Array, Array]:
    """Get the full dataset"""
    images = jnp.array(partitioner.images)
    if hasattr(partitioner, "transform"):
        transforms = partitioner.get_transform()
        images = jnp.array([transforms(img).permute(1, 2, 0).numpy() for img in images])
    labels = partitioner.labels
    labels = jax.nn.one_hot(labels, jnp.max(labels) + 1)
    rnd_perm = np.random.permutation(np.arange(images.shape[0]))
    images = images[rnd_perm]
    labels = labels[rnd_perm]
    return (images, labels)


def get_fewshot_subsets(
    args, n_samples, rnd_seed
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    train_partitioner = DataPartitioner(
        dataset=args.dataset,
        data_path=args.data_path,
        n_samples=n_samples,
        distribution=args.distribution,
        seed=rnd_seed,
        min_samples=args.min_samples,
        train=True,
    )
    """
    val_partitioner = DataPartitioner(
        dataset=args.dataset,
        data_path=args.data_path,
        n_samples=n_samples,
        distribution=args.distribution,
        seed=rnd_seed,
        min_samples=args.min_samples,
        train=True,
    )
    """
    if n_samples:
        # get a subset of the data with M samples per class
        images, labels = train_partitioner.get_subset()
        train_set, val_set = train_partitioner.create_splits(images, labels)
    else:
        train_set = get_full_dataset(train_partitioner)
        val_partitioner = DataPartitioner(
                        dataset=args.dataset,
                        data_path=args.data_path,
                        n_samples=n_samples,
                        distribution=args.distribution,
                        seed=rnd_seed,
                        min_samples=args.min_samples,
                        train=False,
        )
        val_set = get_full_dataset(val_partitioner)
    return train_set, val_set


def get_class_distribution(T: int, k: int = 3, p: float = .7) -> Array:
    """With probabilities $(p/k)$ and $(1-p)/(T-k)$ sample $k$ frequent and $T-k$ rare classes respectively."""
    distribution = np.zeros(T)
    p_k = (p/k)
    q_k = (1-p)/(T-k)
    frequent_classes = np.random.choice(T, size=k, replace=False)
    rare_classes = np.asarray(list(set(range(T)).difference(list(frequent_classes))))
    distribution[frequent_classes] += p_k
    distribution[rare_classes] += q_k
    return distribution


def sample_instances(n_classes: int, n_totals: int) -> Array:
    class_distribution = get_class_distribution(n_classes)
    sample = np.random.choice(n_classes, size=n_totals, replace=True, p=class_distribution)
    sample = add_remainder(sample, n_classes)
    return sample


def add_remainder(sample: np.ndarray, n_classes: int) -> np.ndarray:
    remainder = np.array(
        [y for y in np.arange(n_classes) if y not in np.unique(sample)]
    )
    sample = np.hstack((sample, remainder))
    return sample


def get_histogram(sample: np.ndarray, min_samples: int) -> np.ndarray:
    _, hist = zip(*sorted(Counter(sample).items(), key=lambda kv: kv[0], reverse=False))
    hist = np.array(hist)
    # guarantee that there are at least C (= min_samples) examples per class
    hist = np.where(hist < min_samples, hist + abs(hist - min_samples), hist)
    return hist


def get_subset(y, hist):
    subset = []
    for k, freq in enumerate(hist):
        subset.extend(
            np.random.choice(np.where(y == k)[0], size=freq, replace=False).tolist()
        )
    subset = np.random.permutation(subset)
    return subset


def sample_subset(
    X: np.ndarray, 
    y: np.ndarray, 
    N: int, 
    C: int,
) -> Tuple[np.ndarray, np.ndarray]:
    M = len(np.unique(y))
    sample = sample_instances(M, N)
    hist = get_histogram(sample, C)
    subset = get_subset(y, hist)
    X_prime = X[subset]
    y_prime = y[subset]
    y_prime = jax.nn.one_hot(y_prime, M)
    return (X_prime, y_prime)


def get_class_weights(M: int, K: int, C: int) -> np.ndarray:
    """Compute class weights to calculate weighted cross-entropy error."""
    N = M * K
    sample = sample_instances(K, N)
    hist = get_histogram(sample, C)
    q_prime = hist / hist.sum()  # empirical class distribution (different from q)
    w = q_prime ** (-1)
    # smooth class weights to avoid exploding gradient/numerical overflow issues
    w = (w / w.sum()) * K
    return jax.device_put(w)
