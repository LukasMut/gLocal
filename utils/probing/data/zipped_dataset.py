import os
from typing import List, Tuple

import h5py
import torch

Tensor = torch.Tensor


class ZippedData(torch.utils.data.Dataset):
    def __init__(
        self,
        triplets: List[List[int]],
        n_objects: int,
        features_root: str,
        split: str,
        format: str,
        device: str = "cuda",
    ) -> None:
        super(ZippedData, self).__init__()
        self.triplets = torch.tensor(triplets).type(torch.LongTensor)
        self.identity = torch.eye(n_objects)
        self.root = features_root
        self.split = split
        self.format = format
        self.device = device
        self.num_triplets = self.triplets.shape[0]
        self.load_data()

    def get_features(self, index: int) -> Tensor:
        if self.format == "pt":
            features = torch.load(self.feature_order[index], map_location=self.device)
        elif self.format == "hdf5":
            features = torch.from_numpy(self.h5py_view[self.h5py_key][index]).to(
                torch.float32
            )
        return features

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if self.num_triplets > self.num_features:
            triplet = self.encode_as_onehot(self.triplets[index])
            features = self.get_features(index % self.num_features)
        elif self.num_triplets < self.num_features:
            triplet = self.encode_as_onehot(self.triplets[index % self.num_triplets])
            features = self.get_features(index)
        else:
            triplet = self.encode_as_onehot(self.triplets[index])
            features = self.get_features(index)
        return triplet, features

    def encode_as_onehot(self, triplet: Tensor) -> Tensor:
        """Encode a triplet of indices as a matrix of three one-hot-vectors."""
        return self.identity[triplet, :]

    def load_data(self) -> None:
        """Load features into memory."""
        if self.format == "hdf5":
            self.h5py_view = h5py.File(
                os.path.join(self.root, self.split, "features.hdf5"), "r"
            )
            self.h5py_key = list(self.h5py_view.keys()).pop()
            self.num_features = self.h5py_view[self.h5py_key].shape[0]
        elif self.format == "pt":
            self.feature_order = sorted(
                [
                    os.path.join(self.root, self.split, f.name)
                    for f in os.scandir(os.path.join(self.root, self.split))
                    if f.name.endswith("pt")
                ]
            )
            self.num_features = len(self.feature_order)

    def __len__(self) -> int:
        if self.num_triplets > self.num_features:
            self.length = self.num_triplets
        else:
            self.length = self.num_features
        return self.length
