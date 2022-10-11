#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["DataLoader"]

import copy
import math
import random
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Iterator, List, Tuple

import numpy as np
import torch
from ml_collections import config_dict

FrozenDict = config_dict.FrozenConfigDict
Tensor = torch.Tensor
Array = np.ndarray

@dataclass
class DataLoader:
    data: Tuple[Tensor]
    data_config: FrozenDict
    model_config: FrozenDict
    seed: int
    train: bool = True
    class_subset: List[int] = None

    def __post_init__(self):
        # seed random number generator
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.X = torch.from_numpy(np.asarray(self.data[0]))
        self.class_subset = copy.deepcopy(self.class_subset)

        if self.data_config.name.endswith("mnist"):
            self.X = torch.unsqueeze(self.X, dim=-1)

        self.X = self.X.permute(0, 3, 1, 2)
        self.y = torch.from_numpy(copy.deepcopy(self.data[1]))

        self.dataset = list(zip(self.X, self.y))
        self.num_batches = math.ceil(len(self.dataset) / self.data_config.batch_size)

        if self.data_config.sampling == "uniform":
            num_classes = self.y.shape[-1]
            self.y_flat = torch.nonzero(self.y)[:, 1]

            occurrences = dict(
                sorted(Counter(self.y_flat.tolist()).items(), key=lambda kv: kv[0])
            )
            self.hist = np.array(list(occurrences.values()))
            self.p = self.hist / self.hist.sum()
            self.temperature = 0.1

            if self.class_subset:
                self.classes = torch.tensor(self.class_subset)
            else:
                self.classes = torch.arange(num_classes)
        else:
            self.remainder = len(self.dataset) % self.data_config.batch_size

        self.create_functions()

    def create_functions(self) -> None:
        def unzip_pairs(
            dataset: Tuple[Tensor, Tensor],
            subset: range,
            sampling: str,
            train: bool,
            random_order=None,
        ) -> Tuple[Tensor, Tensor]:
            """Create tuples of data pairs (X, y)."""
            if sampling == "standard" and train:
                assert not isinstance(
                    random_order, type(None)
                ), "\nThe order in which examples are presented to the model should be randomly permuted.\n"
            X, y = zip(*[dataset[random_order[i]] if (sampling == 'standard' and train) else dataset[i] for i in subset])
            X = torch.stack(X, dim=0)
            y = torch.stack(y, dim=0)
            return (X, y)

        self.unzip_pairs = partial(unzip_pairs, self.dataset)

    def stepping(self, random_order: Tensor) -> Tuple[Tensor, Tensor]:
        """Step over the entire training data in mini-batches of size B."""
        for i in range(self.num_batches):
            if self.remainder != 0 and i == int(self.num_batches - 1):
                subset = range(
                    i * self.data_config.batch_size,
                    i * self.data_config.batch_size + self.remainder,
                )
            else:
                subset = range(
                    i * self.data_config.batch_size,
                    (i + 1) * self.data_config.batch_size,
                )
            X, y = self.unzip_pairs(
                subset=subset,
                sampling=self.data_config.sampling,
                train=self.train,
                random_order=random_order,
            )
            yield (X, y)

    def get_random_subset(self, sample) -> List[int]:
        subset = []
        for cls in sample:
            candidates = torch.where(self.y_flat == cls)[0]
            idx = candidates[torch.randperm(candidates.shape[0])[0]]
            subset.append(idx)
        return subset

    def sample_main_batch(self, q=None) -> Tuple[Tensor, Tensor]:
        sample = np.random.choice(self.classes, size=self.data_config.batch_size, p=q)
        subset = self.get_random_subset(sample)
        X, y = self.unzip_pairs(
            subset=subset,
            sampling=self.data_config.sampling,
            train=self.train,
        )
        return (X, y)

    def softmax(self) -> Array:
        return np.exp(self.p / self.temperature) / (np.exp(self.p / self.temperature).sum())

    def main_batch_balancing(self) -> Tuple[Tensor, Tensor]:
        """Sample classes uniformly for each randomly sampled mini-batch."""
        q = self.softmax()
        q = None
        for _ in range(self.num_batches):
            main_batch = self.sample_main_batch(q)
            yield main_batch
        self.temperature += .01

    def __iter__(self) -> Iterator:
        if self.data_config.sampling == "standard":
            if self.train:
                # randomly permute the order of samples in the data (i.e., for each epoch shuffle the data)
                random_order = np.random.permutation(np.arange(len(self.dataset)))
            return iter(self.stepping(random_order))
        else:
            return iter(self.main_batch_balancing())

    def __len__(self) -> int:
        return self.num_batches
