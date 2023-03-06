import os

import h5py
import torch

Tensor = torch.Tensor


class FeaturesPT(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train", device: str = "cuda") -> None:
        super(FeaturesPT, self).__init__()
        self.root = root
        self.split = split
        self.device = torch.device(device)
        self.feature_order = sorted(
            [
                os.path.join(self.root, self.split, f.name)
                for f in os.scandir(os.path.join(self.root, self.split))
                if f.name.endswith("pt")
            ]
        )

    def __getitem__(self, idx: int) -> Tensor:
        return torch.load(self.feature_order[idx], map_location=self.device)

    def __len__(self) -> int:
        return len(self.feature_order)


class FeaturesHDF5(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train") -> None:
        super(FeaturesHDF5, self).__init__()
        self.root = root
        self.split = split
        self.h5py_view = h5py.File(
            os.path.join(self.root, self.split, "features.hdf5"), "r"
        )
        self.h5py_key = list(self.h5py_view.keys()).pop()
        # features = torch.from_numpy(self.h5py_view[self.h5py_key][:])
        # self.features = features.to(torch.float32)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.from_numpy(self.h5py_view[self.h5py_key][idx]).to(torch.float32)
        # return self.features[idx]

    def __len__(self) -> int:
        return self.h5py_view[self.h5py_key].shape[0] 
