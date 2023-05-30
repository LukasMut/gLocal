from multiprocessing.sharedctypes import Value
import os
import pickle
import numpy as np
import zipfile
import utils
import torch
import pickle

Array = np.ndarray
FILE_FORMATS = [".pkl", ".npz"]


class THINGSFeatureTransform(object):
    def __init__(
            self,
            source: str = "custom",
            model_name: str = "clip_ViT-B/16",
            module: str = "penultimate",
            path_to_transform: str = "/home/space/datasets/things/transforms/transforms_without_norm.pkl",
            things_embeddings_path: str = "/home/space/datasets/things/embeddings/model_features_per_source.pkl",
            archive_path=None,
            device='cpu'
    ):
        self.source = source
        self.model_name = model_name
        self.module = module
        self.new_transform = path_to_transform.endswith('.npz')

        if archive_path:
            archive = zipfile.ZipFile(archive_path, 'r')
            f = archive.open(path_to_transform)
            self.transform = np.load(f, mmap_mode='r')
        else:
            self._load_transform(path_to_transform)

        if "mean" in self.transform and "std" in self.transform:
            self.things_mean = self.transform["mean"]
            self.things_std = self.transform["std"]
        else:
            features_things = utils.evaluation.load_features(path=things_embeddings_path)
            self.things_mean = np.mean(features_things[source][model_name][self.module])
            self.things_std = np.std(features_things[source][model_name][self.module])
        self.things_mean = torch.tensor(self.things_mean).to(device)
        self.things_std = torch.tensor(self.things_std).to(device)

        if self.new_transform:
            self.variables = {}
            for key in ["weights", "bias"]:
                if key in self.transform:
                    self.variables[key] = torch.tensor(self.transform[key]).to(device)
        else:
            self.transform = torch.tensor(self.transform).to(device)

    def _load_transform(self, path_to_transform: str) -> None:
        assert os.path.isfile(
            path_to_transform
        ), "\nThe provided path does not point to a file.\n"
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
        if self.new_transform:
            if "weights" in self.variables:
                features = features @ self.variables["weights"]
                if "bias" in self.variables:
                    features += self.variables["bias"]
            else:
                raise KeyError('\nWeights not found in transform.\n')
        else:
            if (self.transform.shape[0] != self.transform.shape[1]):
                weights = self.transform[:,:-1]
                bias = self.transform[:,-1]
                features = features @ weights + bias
            else:
                features = features @ self.transform
        return features
