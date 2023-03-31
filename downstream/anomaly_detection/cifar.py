from downstream.anomaly_detection.base import BaseADSet
from typing import List
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from .utils import get_target_label_idx
from data.cifar import CIFAR100Coarse


class ADCIFAR10(BaseADSet):
    def __init__(self, data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

    def create_datasets(self, train_transform, test_transform):
        train = CIFAR10(root=self.data_dir,
                        train=True,
                        download=True,
                        transform=train_transform)

        test = CIFAR10(root=self.data_dir,
                       train=False,
                       download=True,
                       transform=test_transform)
        return train, test

    def reduce_train(self, train_embeddings, normal_cls):
        dataset_targets = self._train.targets
        train_idx_normal = get_target_label_idx(dataset_targets, np.array([normal_cls]))
        return train_embeddings[train_idx_normal]

    def reduce_test(self, test_embeddings, normal_cls):
        dataset_targets = np.array(self._test.targets)
        return test_embeddings, dataset_targets != normal_cls


class ADCIFAR100(BaseADSet):
    def __init__(self, data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

    def create_datasets(self, train_transform, test_transform):
        train = CIFAR100(root=self.data_dir,
                         train=True,
                         download=True,
                         transform=train_transform)

        test = CIFAR100(root=self.data_dir,
                        train=False,
                        download=True,
                        transform=test_transform)
        return train, test

    def reduce_train(self, train_embeddings, normal_cls):
        dataset_targets = self._train.targets
        train_idx_normal = get_target_label_idx(dataset_targets, np.array([normal_cls]))
        return train_embeddings[train_idx_normal]

    def reduce_test(self, test_embeddings, normal_cls):
        dataset_targets = np.array(self._test.targets)
        return test_embeddings, dataset_targets != normal_cls


class ADCIFAR100Shift(BaseADSet):

    def __init__(self, train_indices, data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.train_indices = train_indices

    def create_datasets(self, train_transform, test_transform):

        train = CIFAR100Coarse(root=self.data_dir,
                               train=True,
                               download=True,
                               transform=train_transform)

        test = CIFAR100Coarse(root=self.data_dir,
                              train=False,
                              download=True,
                              transform=test_transform)
        return train, test

    def reduce_train(self, train_embeddings, normal_cls):
        fine_classes = np.argwhere(self._train.coarse_labels == normal_cls)
        train_idx_normal = get_target_label_idx(self._train.fine_targets,
                                                np.array(fine_classes[self.train_indices]))
        return train_embeddings[train_idx_normal]

    def reduce_test(self, test_embeddings, normal_cls):
        fine_classes = np.argwhere(self._test.coarse_labels == normal_cls)
        all_indices = []
        for idx in list(range(100)):
            if idx not in fine_classes[self.train_indices]:
                all_indices.append(idx)
        indices = get_target_label_idx(self._test.fine_targets, np.array(all_indices))
        return test_embeddings[indices], self._test.targets[indices] != normal_cls
