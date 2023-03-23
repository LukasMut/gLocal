from downstream.anomaly_detection.base import BaseADSet
from typing import List
from torchvision.datasets import Flowers102
import numpy as np
from torch.utils.data import Subset
from .utils import get_target_label_idx


class ADFlowers(BaseADSet):

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

        train = Flowers102(root=self.data_dir,
                           split='train',
                           download=True,
                           transform=train_transform,
                           target_transform=target_transform)
        train.targets = train._labels
        train = self.reduce_subset(train)

        test = Flowers102(root=self.data_dir,
                          split='test',
                          download=True,
                          transform=test_transform,
                          target_transform=target_transform)
        test.targets = test._labels
        return train, test

    def reduce_subset(self, dataset):
        dataset_targets = dataset.targets
        train_idx_normal = get_target_label_idx(dataset_targets, np.array(self.normal_classes))
        return Subset(dataset, train_idx_normal)
