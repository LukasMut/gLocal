#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pickle
import sys

import numpy as np
import jax.numpy as jnp

from typing import List, Tuple

MEANS = [0.49140, 0.48216, 0.44653]
STDS = [0.24703, 0.24349, 0.26159]


def unpickle(file) -> dict:
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def normalize_images(images: jnp.ndarray, means: List[float], stds: List[float]) -> jnp.ndarray:
    images = images / 255
    for c, (mean, std) in enumerate(zip(means, stds)):
        images[:, :, :, c] -= mean
        images[:, :, :, c] /= std
    return jnp.array(images)


def preproc_cifar_10(root: str, train: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    prefix = 'data' if train else 'test'
    regex = f'(?=^{prefix})(?=.*batch)(?=.*\d*$)'
    images, labels = [], []
    for d in os.scandir(root):
        if re.compile(regex).search(d.name):
            data_batch = unpickle(os.path.join(root, d.name))
            image_batch = data_batch[b'data']
            label_batch = data_batch[b'labels']
            image_batch = reshape_images(image_batch)
            label_batch = jnp.array(label_batch)
            images.append(image_batch)
            labels.append(label_batch)
    images = np.vstack(images)
    labels = jnp.hstack(labels)
    if not train:
        images = normalize_images(images=images, means=MEANS, stds=STDS)
    return (images, labels)


def reshape_images(images: np.ndarray) -> jnp.ndarray:
    return jnp.apply_along_axis(reshape_image, axis=1, arr=images)


def reshape_image(image: np.ndarray) -> jnp.ndarray:
    """Bring images into the right format."""
    C = 3
    row_entries = 32
    chan_entries = row_entries ** 2
    channels = [image[(i * chan_entries):(i + 1) * chan_entries]
                for i in range(C)]
    channels = list(map(lambda x: x.reshape(
        row_entries, row_entries), channels))
    image = jnp.stack(channels, axis=-1)
    return image


def save_data(dataset: Tuple[jnp.ndarray, jnp.ndarray], split: str = 'training') -> None:
    with open(os.path.join(out_path, f'{split}.npz'), 'wb') as f:
        np.savez_compressed(f, data=dataset[0], labels=dataset[1])


if __name__ == '__main__':
    root = sys.argv[1] # path/to/unprocessed/cifar10/images
    split = sys.argv[2] # training or test
    out_path = sys.argv[3] # path/to/processed/cifar10/images

    cifar10_data = preproc_cifar_10(
        root, train=True if re.search(r'^train', split) else False)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    save_data(cifar10_data, split)
