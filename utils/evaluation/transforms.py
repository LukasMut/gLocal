__all__ = ["GlobalTransform", "GlocalTransform"]

import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray

FILE_FORMATS = ["pkl", "npz"]


@dataclass
class GlobalTransform:
    source: str = "custom"
    model_name: str = "clip_ViT-B/16"
    module: str = "penultimate"
    path_to_transform: str = (
        "/home/space/datasets/things/transforms/transforms_without_norm.pkl"
    )
    path_to_features: str = (
        "/home/space/datasets/things/probing/embeddings/features.pkl"
    )

    def __post_init__(self) -> None:
        self._load_transform(self.path_to_transform)
        self._load_features(self.path_to_features)

    def _load_features(self, path_to_features: str) -> None:
        assert os.path.isfile(
            path_to_features
        ), "\nThe provided path does not point to a file.\nChange path.\n"
        with open(path_to_features, "rb") as f:
            things_features = pickle.load(f)
        things_features = things_features[self.source][self.model_name][self.module]
        self.things_mean = things_features.mean()
        self.things_std = things_features.std()

    def _load_transform(self, path_to_transform: str) -> None:
        assert os.path.isfile(
            path_to_transform
        ), f"\nThe provided path does not point to a valid file:{path_to_transform}\n"
        if path_to_transform.endswith("pkl"):
            with open(path_to_transform, "rb") as f:
                transforms = pickle.load(f)
            self.transform = transforms[self.source][self.model_name][self.module]
        elif path_to_transform.endswith("npz"):
            self.transform = np.load(path_to_transform)
        else:
            raise ValueError(
                f"\nThe provided file does not have a valid format. Valid formats are: {FILE_FORMATS}\n"
            )

    def transform_features(self, features: Array) -> Array:
        features = (features - self.things_mean) / self.things_std
        if "weights" in self.transform:
            features = features @ self.transform["weights"]
            if "bias" in self.transform:
                features += self.transform["bias"]
        elif self.transform.shape[0] != self.transform.shape[1]:
            weights = self.transform[:, :-1]
            bias = self.transform[:, -1]
            features = features @ weights + bias
        else:
            features = features @ self.transform
        return features


@dataclass
class GlocalTransform:
    root: str = "/home/space/datasets/things/probing/"
    source: str = "custom"
    model: str = "clip_RN50"
    module: str = "penultimate"
    optim: str = "sgd"
    eta: float = 0.001
    lmbda: float = 1.0
    alpha: float = 0.25
    tau: float = 0.1
    contrastive_batch_size: float = 1024
    adversarial: bool = False

    def __post_init__(
        self,
    ) -> None:
        args = [
            self.root,
            self.source,
            self.model,
            self.module,
            self.optim.lower(),
            self.eta,
            self.lmbda,
            self.alpha,
            self.tau,
            self.contrastive_batch_size,
        ]
        if self.adversarial:
            args.append("adversarial")
        path_to_transform = os.path.join(*[str(arg) for arg in args])
        self.transform = self._load_transform(path_to_transform)

    @staticmethod
    def _load_transform(path_to_transform: str) -> Any:
        path_to_transform_file = os.path.join(path_to_transform, "transform.npz")
        assert os.path.isfile(
            path_to_transform_file
        ), f"\nThe provided path does not point to a valid file:{path_to_transform_file}\n"
        transform = np.load(path_to_transform_file)
        return transform

    def transform_features(self, features: Array) -> Array:
        things_mean = self.transform["mean"]
        things_std = self.transform["std"]
        features = (features - things_mean) / things_std
        if "weights" in self.transform:
            features = features @ self.transform["weights"]
            if "bias" in self.transform:
                features += self.transform["bias"]
        return features
