import os
import torch
import numpy as np
from typing import Optional, Any, Tuple
from torchvision.datasets import ImageNet

Array = np.ndarray
Tensor = torch.Tensor


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


def get_target_label_idx(labels: np.ndarray, targets: np.ndarray):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()
