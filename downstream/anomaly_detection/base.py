from torch.utils.data import DataLoader


class BaseADSet:
    def __init__(self, transform, batch_size=64, num_workers=0):
        self.normal_label = 0
        self.anomalous_label = 1

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

    def reduce_train(self, train_embeddings, normal_cls):
        pass

    def reduce_test(self, test_embeddings, normal_cls):
        pass
