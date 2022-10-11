
import torch 
import torchvision
from typing import Any, Tuple, List
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T


Tensor = torch.Tensor


class Rotations(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        transforms = torch.nn.Sequential(T.RandomRotation(90))
        self.transforms = torch.jit.script(transforms)

    def forward(self, x: Tensor) -> Tensor:
        return self.transforms(x)


class HorizontalFlips(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        transforms = torch.nn.Sequential(T.RandomHorizontalFlip())
        self.transforms = torch.jit.script(transforms)

    def forward(self, x: Tensor) -> Tensor:
        return self.transforms(x)


class VerticalFlips(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        transforms = torch.nn.Sequential(T.RandomVerticalFlip())
        self.transforms = torch.jit.script(transforms)

    def forward(self, x: Tensor) -> Tensor:
        return self.transforms(x)


class AugmentationHead(nn.Module):

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()

        self.rotation_head = nn.Linear(input_dim, num_classes)
        self.hflip_head = nn.Linear(input_dim, num_classes)
        self.vflip_head = nn.Linear(input_dim, num_classes)

    
    def forward(self, x: List[Tensor]) -> Tuple[Tensor]:
        out_r = self.rotation_head(x[0])
        out_h = self.hflip_head(x[1])
        out_v = self.vflip_head(x[2])
        return out_r, out_h, out_v


class CLFHead(nn.Module):

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()

        self.clf_head = nn.Linear(input_dim, num_classes)

    
    def forward(self, x: Tensor) -> Tensor:
        out = self.clf_head(x)
        return out


class AugmentationNet(nn.Module):

    def __init__(self, backbone: str, num_classes: int, grayscale: bool) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.grayscale = grayscale
        self.load_feature_extractor()
        self._replace_layers()
        self.transforms = [Rotations(), HorizontalFlips(), VerticalFlips()]
        self.augmentation_head = AugmentationHead(self.feature_dim, self.num_classes)
        self.clf = CLFHead(self.feature_dim, self.num_classes)
    
    def forward(self, x: Tensor, train: bool = True) -> Tensor:
        x = x.type(torch.FloatTensor)
        x_o = self.extractor(x)
        out_o = self.clf(x_o)
        if train:
            views = [self.extractor(transform(x)) for transform in self.transforms]
            out_r, out_h, out_v = self.augmentation_head(views)
            out = [out_o, out_r, out_h, out_v]
        else:
            out = out_o
        return out
    
    def _replace_layers(self) -> None:
        if self.backbone == 'alexnet':
            self.feature_dim = self.extractor.classifier[6].in_features
            self.extractor.classifier[6] = nn.Identity()
            if self.grayscale:
                self.extractor.features[0] = nn.Conv2d(1, 64, 7, 1, 1, bias=False)
        else:
            self.feature_dim = self.extractor.fc.in_features
            self.extractor.fc = nn.Identity()
            if self.grayscale:
                self.extractor.conv1 = nn.Conv2d(1, 64, 7, 1, 1, bias=False)


    def load_feature_extractor(self) -> Any:
        """Get a feature extractor from from <torchvision>."""
        if hasattr(torchvision.models, self.backbone):
            extractor = getattr(torchvision.models, self.backbone)
            self.extractor = extractor(weights=None)
        else:
            raise ValueError(f'\n{self.backbone} cannot be found among all <torchvision> models. Please choose a different feature extractor.\n')
    