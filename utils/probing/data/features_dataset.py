import os

import torch

Tensor = torch.Tensor


class Features(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str = "train", device: str = "cuda") -> None:
        super(Features, self).__init__()
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
