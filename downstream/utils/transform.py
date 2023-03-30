from multiprocessing.sharedctypes import Value
import os
import pickle
import numpy as np
import zipfile

Array = np.ndarray
FILE_FORMATS = [".pkl", ".npz"]


class THINGSFeatureTransform(object):
    def __init__(
            self,
            source: str = "custom",
            model_name: str = "clip_ViT-B/16",
            module: str = "penultimate",
            path_to_transform: str = "/home/space/datasets/things/transforms/transforms_without_norm.pkl",
            archive_path=None
    ):
        self.source = source
        self.model_name = model_name
        self.module = module
        if archive_path:
            archive = zipfile.ZipFile(archive_path, 'r')
            f = archive.open(path_to_transform)
            self.transform = np.load(f, mmap_mode='r')
        else:
            self._load_transform(path_to_transform)

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
        things_mean = self.transform["mean"]
        things_std = self.transform["std"]
        features = (features - things_mean) / things_std
        if "weights" in self.transform:
            features = features @ self.transform["weights"]
            if "bias" in self.transform:
                features += self.transform["bias"]
        return features
