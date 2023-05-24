from torch.utils.data import DataLoader
import numpy as np
from .utils import get_target_label_idx


class BaseADSet:
    def __init__(self, transform, mode='ovr', batch_size=64, num_workers=0):
        self.normal_label = 0
        self.anomalous_label = 1

        self.mode = mode

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._transform = transform

    def setup(self):
        self._train, self._test = self.create_datasets(train_transform=self._transform, test_transform=self._transform)

    def create_datasets(self, train_transform, test_transform):
        raise NotImplementedError('Subclass has to implement this method')

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self._batch_size,
                          num_workers=self._num_workers, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self._batch_size, num_workers=self._num_workers)

    def reduce_train(self, train_embeddings, cls):
        dataset_targets = self._train.targets
        if self.mode == 'ovr':
            train_idx_normal = get_target_label_idx(dataset_targets, np.array([cls]))
        elif self.mode == 'rvo':
            labels = np.unique(dataset_targets)
            normal_labels = labels.tolist()
            normal_labels.remove(cls)
            train_idx_normal = get_target_label_idx(dataset_targets, np.array(normal_labels))
        else:
            raise ValueError('unknown mode')
        return train_embeddings[train_idx_normal]

    def reduce_test(self, test_embeddings, cls):
        dataset_targets = np.array(self._test.targets)
        if self.mode == 'ovr':
            labels = dataset_targets != cls
        elif self.mode == 'rvo':
            labels = dataset_targets == cls
        else:
            raise ValueError('unknown mode')
        return test_embeddings, labels

    def is_embedded(self):
        return False
