from downstream.anomaly_detection.base import BaseADSet
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from .utils import get_target_label_idx
from data.cifar import CIFAR100Coarse
from torch.utils.data import ConcatDataset


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

    def cache_name(self):
        return 'cifar10'

    def class_names(self):
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']


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

    def cache_name(self):
        return 'cifar100'

    def class_names(self):
        return ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
                'mountain',
                'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
                'pine_tree',
                'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
                'tiger', 'tractor',
                'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


class ADCIFAR100Coarse(BaseADSet):
    def __init__(self, data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

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

    def cache_name(self):
        return 'cifar100'

    def class_names(self):
        return ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                'large man-made outdoor things',
                'large natural outdoor scenes', 'large omnivores and herbivores', 'medium-sized mammals',
                'non-insect invertebrates',
                'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']


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

    def reduce_train(self, train_embeddings, cls):
        fine_classes = np.argwhere(self._train.coarse_labels == cls)
        train_idx_normal = get_target_label_idx(self._train.fine_targets,
                                                np.array(fine_classes[self.train_indices]))
        return train_embeddings[train_idx_normal]

    def reduce_test(self, test_embeddings, cls):
        fine_classes = np.argwhere(self._test.coarse_labels == cls)
        all_indices = []
        for idx in list(range(100)):
            if idx not in fine_classes[self.train_indices]:
                all_indices.append(idx)
        indices = get_target_label_idx(self._test.fine_targets, np.array(all_indices))
        return test_embeddings[indices], self._test.targets[indices] != cls

    def cache_name(self):
        return 'cifar100'

    def class_names(self):
        return ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                'large man-made outdoor things',
                'large natural outdoor scenes', 'large omnivores and herbivores', 'medium-sized mammals',
                'non-insect invertebrates',
                'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']


class ADCIFAR10vs100(BaseADSet):
    def __init__(self, data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

    def create_datasets(self, train_transform, test_transform):
        train = CIFAR10(root=self.data_dir,
                        train=True,
                        download=True,
                        transform=train_transform)

        test_10 = CIFAR10(root=self.data_dir,
                          train=False,
                          download=True,
                          transform=test_transform)
        test_100 = CIFAR100(root=self.data_dir,
                            train=False,
                            download=True,
                            transform=test_transform)

        return train, ConcatDataset([test_10, test_100])

    def reduce_train(self, train_embeddings, cls):
        return train_embeddings

    def reduce_test(self, test_embeddings, cls):
        labels = np.concatenate((np.zeros(10000), np.ones(10000)))
        return test_embeddings, labels

    def cache_name(self):
        return 'cifar10vs100'
