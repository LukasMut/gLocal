from downstream.anomaly_detection.base import BaseADSet
from torchvision.datasets import DTD
import numpy as np
from .utils import get_target_label_idx


class ADDTD(BaseADSet):

    def __init__(self, data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

    def create_datasets(self, train_transform, test_transform):
        train = DTD(root=self.data_dir,
                    split='train',
                    download=True,
                    transform=train_transform)
        train.targets = train._labels
        test = DTD(root=self.data_dir,
                   split='test',
                   download=True,
                   transform=test_transform)
        test.targets = test._labels
        return train, test

    def reduce_train(self, train_embeddings, normal_cls):
        dataset_targets = self._train.targets
        train_idx_normal = get_target_label_idx(dataset_targets, np.array([normal_cls]))
        return train_embeddings[train_idx_normal]

    def reduce_test(self, test_embeddings, normal_cls):
        dataset_targets = np.array(self._test.targets)
        return test_embeddings, dataset_targets != normal_cls
