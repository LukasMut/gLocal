import os
import torch
import numpy as np
from typing import Optional, Any, Tuple, Union, List
from torchvision.datasets import ImageNet
from downstream.anomaly_detection.base import BaseADSet
from .utils import get_target_label_idx
from .breeds_sets import get_breeds_task

Array = np.ndarray
Tensor = torch.Tensor

IMAGENET_ROOT = '../unsup-clever-hans/resources/old/resources/imagenet'


class EmbeddedImageNet(ImageNet):
    def __init__(
            self,
            root: str,
            embedding_root: str,
            split: str = "train",
            device: str = "cpu",
            **kwargs: Any
    ) -> None:
        super(EmbeddedImageNet, self).__init__(root=root, split=split, **kwargs)
        self.device = torch.device(device)
        self.feature_order = sorted(
            [
                os.path.join(embedding_root, self.split, f.name)
                for f in os.scandir(os.path.join(embedding_root, self.split))
                if f.name.endswith("pt")
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        _, target = self.samples[index]
        sample = torch.load(self.feature_order[index], map_location=torch.device(self.device))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ADImageNet30(BaseADSet):
    ad_classes = ['acorn', 'airliner', 'ambulance', 'American alligator', 'banjo', 'barn', 'bikini', 'digital clock',
                  'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand piano', 'hotdog', 'hourglass', 'manhole cover',
                  'mosque', 'nail', 'parking meter', 'pillow', 'revolver', 'dial telephone', 'schooner',
                  'snowmobile', 'soccer ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']

    def __init__(self, embeddings_root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings_root = embeddings_root

    def create_datasets(self, train_transform, test_transform):
        train = EmbeddedImageNet(root=IMAGENET_ROOT,
                                 embedding_root=self.embeddings_root,
                                 split='train',
                                 transform=train_transform)

        ad_labels = [train.class_to_idx[c] for c in self.ad_classes]

        subset_indices = get_target_label_idx(train.targets, ad_labels)
        train_subset = torch.utils.data.Subset(train, subset_indices)
        subset_indices = np.array(subset_indices)
        subset_targets = np.array(train.targets)[subset_indices]
        subset_targets = [ad_labels.index(c) for c in subset_targets]
        train_subset.targets = subset_targets

        test = EmbeddedImageNet(root=IMAGENET_ROOT,
                                embedding_root=self.embeddings_root,
                                split='val',
                                transform=test_transform)

        subset_indices = get_target_label_idx(test.targets, ad_labels)
        test_subset = torch.utils.data.Subset(test, subset_indices)
        subset_indices = np.array(subset_indices)
        subset_targets = np.array(test.targets)[subset_indices]
        subset_targets = [ad_labels.index(c) for c in subset_targets]
        test_subset.targets = subset_targets

        return train_subset, test_subset

    def cache_name(self):
        return 'imagenet30'

    def is_embedded(self):
        return True


class ADBreeds(BaseADSet):

    def __init__(self, task_name, embeddings_root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings_root = embeddings_root
        self.task_name = task_name

    def create_datasets(self, train_transform, test_transform):
        train_classes, test_classes, superclass_mapping = get_breeds_task(task_name=self.task_name)

        train = EmbeddedImageNet(root=IMAGENET_ROOT,
                                 embedding_root=self.embeddings_root,
                                 split='train',
                                 transform=train_transform)

        subset_indices = get_target_label_idx(train.targets, train_classes)
        train_subset = torch.utils.data.Subset(train, subset_indices)
        subset_indices = np.array(subset_indices)
        targets = np.array(train.targets)
        subset_targets = targets[subset_indices]
        subset_targets = [superclass_mapping[c] for c in subset_targets]
        train_subset.targets = subset_targets

        test = EmbeddedImageNet(root=IMAGENET_ROOT,
                                embedding_root=self.embeddings_root,
                                split='val',
                                transform=test_transform)

        subset_indices = get_target_label_idx(test.targets, test_classes)
        test_subset = torch.utils.data.Subset(test, subset_indices)
        subset_indices = np.array(subset_indices)
        targets = np.array(test.targets)
        subset_targets = targets[subset_indices]
        subset_targets = [superclass_mapping[c] for c in subset_targets]
        test_subset.targets = subset_targets

        return train_subset, test_subset

    def is_embedded(self):
        return True

    def cache_name(self):
        return f'breeds-{self.task_name}'
