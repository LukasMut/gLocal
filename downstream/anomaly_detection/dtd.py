from downstream.anomaly_detection.base import BaseADSet
from torchvision.datasets import DTD


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

    def cache_name(self):
        return 'dtd'
