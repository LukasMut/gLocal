from downstream.anomaly_detection.base import BaseADSet
from typing import List
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from torch.utils.data import Subset
from .utils import get_target_label_idx
from data.cifar import CIFAR100Coarse


class ADCIFAR10(BaseADSet):
    """
    Cifar-10 One vs Rest
    """

    def __init__(self, normal_classes: List[int], data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.normal_classes = normal_classes

    def create_datasets(self, train_transform, test_transform):
        # Map all other samples than the normal class to have the anomalous_label
        def target_transform(y):
            if y in self.normal_classes:
                return self.normal_label
            return self.anomalous_label

        train = CIFAR10(root=self.data_dir,
                        train=True,
                        download=True,
                        transform=train_transform,
                        target_transform=target_transform)
        train = self.reduce_subset(train)

        test = CIFAR10(root=self.data_dir,
                       train=False,
                       download=True,
                       transform=test_transform,
                       target_transform=target_transform)
        return train, test

    def reduce_subset(self, dataset):
        # For Cifar-10, we can use the targets to select normal samples
        dataset_targets = dataset.targets
        train_idx_normal = get_target_label_idx(dataset_targets, np.array(self.normal_classes))
        return Subset(dataset, train_idx_normal)


class ADCIFAR100(BaseADSet):
    """
    Cifar-100 One vs Rest
    """

    def __init__(self, normal_classes: List[int], data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.normal_classes = normal_classes

    def create_datasets(self, train_transform, test_transform):
        # Map all other samples than the normal class to have the anomalous_label
        def target_transform(y):
            if y in self.normal_classes:
                return self.normal_label
            return self.anomalous_label

        train = CIFAR100(root=self.data_dir,
                         train=True,
                         download=True,
                         transform=train_transform,
                         target_transform=target_transform)
        train = self.reduce_subset(train)

        test = CIFAR100(root=self.data_dir,
                        train=False,
                        download=True,
                        transform=test_transform,
                        target_transform=target_transform)
        return train, test

    def reduce_subset(self, dataset):
        # For Cifar-100, we can use the targets to select normal samples
        dataset_targets = dataset.targets
        train_idx_normal = get_target_label_idx(dataset_targets, np.array(self.normal_classes))
        return Subset(dataset, train_idx_normal)


class ADCIFAR100Shift(BaseADSet):
    """
    Cifar-100 shifted One vs Rest
    """

    def __init__(self, normal_class: int, train_indices, data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.normal_class = normal_class
        self.train_indices = train_indices

    def create_datasets(self, train_transform, test_transform):
        # Map all other samples than the normal class to have the anomalous_label
        def target_transform(y):
            if y == self.normal_class:
                return self.normal_label
            return self.anomalous_label

        train = CIFAR100Coarse(root=self.data_dir,
                               train=True,
                               download=True,
                               transform=train_transform,
                               target_transform=target_transform)
        train = self.reduce_train_subset(train, indices=self.train_indices)

        test = CIFAR100Coarse(root=self.data_dir,
                              train=False,
                              download=True,
                              transform=test_transform,
                              target_transform=target_transform)
        test = self.reduce_test_subset(test, exclude_indices=self.train_indices)
        return train, test

    def reduce_train_subset(self, dataset, indices):
        fine_classes = np.argwhere(dataset.coarse_labels == self.normal_class)
        train_idx_normal = get_target_label_idx(dataset.fine_targets, np.array(fine_classes[indices]))
        return Subset(dataset, train_idx_normal)

    def reduce_test_subset(self, dataset, exclude_indices):
        fine_classes = np.argwhere(dataset.coarse_labels == self.normal_class)
        all_indices = []
        for idx in list(range(100)):
            if idx not in fine_classes[exclude_indices]:
                all_indices.append(idx)
        train_idx_normal = get_target_label_idx(dataset.fine_targets, np.array(all_indices))
        return Subset(dataset, train_idx_normal)
