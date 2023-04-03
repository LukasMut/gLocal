from multiprocessing.sharedctypes import Value
import os
import pickle
import numpy as np
import zipfile
import utils

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

        if "mean" in self.transform and "std" in self.transform:
            self.things_mean = self.transform["mean"]
            self.things_std = self.transform["std"]
        else:
            features_things = utils.evaluation.load_features(path=things_embeddings_path)
            self.things_mean = np.mean(
                features_things[source][model_name][self.module],
                # axis=0,
            )
            self.things_std = np.std(
                features_things[source][model_name][self.module],
                # axis=0,
            )

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
        if "weights" in self.transform:
            features = features @ self.transform["weights"]
            if "bias" in self.transform:
                features += self.transform["bias"]
        return features
