from typing import Optional

from torchvision.datasets import CIFAR100, DTD, SUN397

from main_fewshot import Array


def load_dataset(
    name: str,
    data_dir: str,
    train: bool,
    transform=None,
    embeddings: Optional[Array] = None,
):
    if name == "cifar100":
        dataset_class = CIFAR100
        if embeddings is not None:
            dataset_class = embed_dataset(dataset_class, embeddings)
        dataset = dataset_class(
            root=data_dir,
            train=train,
            download=True,
            transform=transform,
        )
    elif name == "DTD":
        dataset_class = DTD
        if embeddings is not None:
            dataset_class = embed_dataset(dataset_class, embeddings)
        dataset = dataset_class(
            root=data_dir,
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    elif name == "SUN397":
        dataset_class = SUN397
        if embeddings is not None:
            dataset_class = embed_dataset(dataset_class, embeddings)
        dataset = dataset_class(
            root=data_dir,
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    else:
        raise ValueError("\nUnknown dataset\n")

    return dataset


def embed_dataset(dataset, embeddings):
    """ Wraps a dataset such that it ueses the given embeddings as features."""
    def __getitem__(self, idx):
        if hasattr(self, "targets"):
            label = self.targets[idx]
        else:
            label = self._labels[idx]

        if hasattr(self, "_image_files"):
            embedding = embeddings[str(self._image_files[idx])]
        else:
            embedding = embeddings[idx]

        if self.target_transform:
            label = self.target_transform(label)
        return embedding, label

    return type(
        dataset.__name__,
        (dataset, ),
        {
            "__getitem__": __getitem__,
        },
    )
