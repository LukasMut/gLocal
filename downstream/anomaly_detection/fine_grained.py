from downstream.anomaly_detection.base import BaseADSet
from torchvision.datasets import Flowers102
import numpy as np
from .utils import get_target_label_idx
from .cub import Cub2011


class ADFlowers(BaseADSet):

    def __init__(self, data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

    def create_datasets(self, train_transform, test_transform):
        train = Flowers102(root=self.data_dir,
                           split='train',
                           download=True,
                           transform=train_transform)
        train.targets = train._labels

        test = Flowers102(root=self.data_dir,
                          split='test',
                          download=True,
                          transform=test_transform)
        test.targets = test._labels
        return train, test

    def cache_name(self):
        return 'flowers'


class ADCUB2011(BaseADSet):

    def __init__(self, data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

    def create_datasets(self, train_transform, test_transform):
        train = Cub2011(root=self.data_dir,
                        train=True,
                        download=True,
                        transform=train_transform)

        test = Cub2011(root=self.data_dir,
                       train=False,
                       download=True,
                       transform=test_transform)
        return train, test

    def cache_name(self):
        return 'cub-2011'
