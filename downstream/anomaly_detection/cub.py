import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import os.path as pt
from torchvision.datasets.utils import download_and_extract_archive
import sys


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = root
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

    def _load_metadata(self):
        if not pt.exists(os.path.join(self.root, self.base_folder, 'images.txt')):
            return False

        images = pd.read_csv(os.path.join(self.root, self.base_folder, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(
            os.path.join(self.root, self.base_folder, 'image_class_labels.txt'), sep=' ', names=['img_id', 'target']
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, self.base_folder, 'train_test_split.txt'), sep=' ',
            names=['img_id', 'is_training_img']
        )

        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')

        if self.train:
            data = data[data.is_training_img == 1]
        else:
            data = data[data.is_training_img == 0]

        data['filepath'] = pt.join(self.root, self.base_folder, 'images') + os.sep + data['filepath']
        self.imgs = self.samples = data.filepath.values
        self.targets = data.target.values - 1  # (1, ..., 200) -> (0, ..., 199)
        self.classes = [f.split('.')[-1] for f in sorted([f for f in set([f.split(os.sep)[-2] for f in self.imgs])])]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        return True

    def _check_integrity(self):
        ret = self._load_metadata()
        if not ret:
            return False

        for fp in self.samples:
            if not pt.isfile(fp):
                print(fp, 'is not found.', file=sys.stderr)
                return False
        return True

    def _download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        else:
            download_and_extract_archive(self.url, pt.join(self.root, ))
            assert self._check_integrity(), 'CUB is corrupted. Please redownload.'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
