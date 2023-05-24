from multiprocessing.sharedctypes import Value
import os
import pickle
import numpy as np

Array = np.ndarray
FILE_FORMATS = [".pkl", ".npz"]


class THINGSFeatureTransform(object):
    def __init__(
        self,
        source: str = "custom",
        model_name: str = "clip_ViT-B/16",
        module: str = "penultimate",
        path_to_transform: str = "/home/space/datasets/things/transforms/transforms_without_norm.pkl",
    ):
        self.source = source
        self.model_name = model_name
        self.module = module
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
        try:
            things_mean = self.transform["mean"]
            things_std = self.transform["std"]
        except:
            with open("../human_alignment/datasets/things/probing/embeddings/features.pkl", "rb") as f:
                things_features = pickle.load(f)
                things_features_current_model = things_features[self.source][self.model_name][self.module]
                things_mean = things_features_current_model.mean()
                things_std = things_features_current_model.std()
        features = (features - things_mean) / things_std
        if "weights" in self.transform:
            print(features.shape)
            print(self.transform["weights"].shape)
            print()
            features = features @ self.transform["weights"]
            if "bias" in self.transform:
                features += self.transform["bias"]
        return features
