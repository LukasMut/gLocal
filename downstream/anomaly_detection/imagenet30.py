import os.path as pt
from typing import List, Tuple
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets.imagenet import META_FILE, parse_train_archive, parse_val_archive, ImageNet
from .base import BaseADSet
from .utils import get_target_label_idx
import torchvision.transforms as transforms

ROOT = pt.join(pt.dirname(__file__), '..')


class ADImageNet(BaseADSet):
    ad_classes = ['acorn', 'airliner', 'ambulance', 'American alligator', 'banjo', 'barn', 'bikini', 'digital clock',
                  'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand piano', 'hotdog', 'hourglass', 'manhole cover',
                  'mosque', 'nail', 'parking meter', 'pillow', 'revolver', 'dial telephone', 'schooner',
                  'snowmobile', 'soccer ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']
    base_folder = 'imagenet'

    def __init__(self, normal_classes: List[int], data_dir: str = './resources/data', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = normal_classes
        self.outlier_classes = list(range(0, 30))
        for cls in self.normal_classes:
            self.outlier_classes.remove(cls)

    def create_datasets(self, train_transform, test_transform):
        target_transform = transforms.Lambda(
            lambda x: self.anomalous_label if x in self.outlier_classes else self.normal_label
        )

        train_set = CustomImageNet(self.data_dir, split='train',
                                   transform=train_transform, target_transform=target_transform)
        train_ad_classes_idx = train_set.get_class_idx(self.ad_classes)
        train_set.targets = [  # t = nan if not in ad_classes else give id in order of ad_classes
            train_ad_classes_idx.index(t) if t in train_ad_classes_idx else np.nan for t in
            train_set.targets
        ]
        dataset_targets = np.asarray(train_set.targets)
        train_idx_normal = get_target_label_idx(dataset_targets, np.array(self.normal_classes))
        train_set = Subset(train_set, train_idx_normal)

        test_set = CustomImageNet(self.data_dir, split='val', transform=test_transform,
                                  target_transform=target_transform)
        test_ad_classes_idx = test_set.get_class_idx(self.ad_classes)
        test_set.targets = [  # t = nan if not in ad_classes else give id in order of ad_classes
            test_ad_classes_idx.index(t) if t in test_ad_classes_idx else np.nan
            for t in test_set.targets
        ]
        test_set = Subset(
            test_set,
            get_target_label_idx(np.asarray(test_set.targets), np.array(list(range(len(self.ad_classes)))))
        )
        return train_set, test_set


class CustomImageNet(ImageNet):

    def get_class_idx(self, classes: List[str]):
        return [self.class_to_idx[c] for c in classes]

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        target = self.targets[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
