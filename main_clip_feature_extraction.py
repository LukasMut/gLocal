import argparse
import os
from typing import List

import numpy as np
import torch
from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader
from thingsvision.utils.storing import save_features
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm

from data import DATASETS, load_dataset

Tensor = torch.Tensor
Array = np.ndarray


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--data_root",
        type=str,
        help="path/to/dataset",
        default="../human_alignment/datasets",
    )
    aa(
        "--datasets",
        type=str,
        nargs="+",
        help="for which datasets to perfrom feature extraction",
        choices=DATASETS,
    )
    aa(
        "--stimulus_set",
        type=str,
        default=None,
        choices=["set1", "set2"],
        help="Similarity judgments of the dataset from King et al. (2019) were collected for two stimulus sets",
    )
    aa(
        "--category",
        type=str,
        default=None,
        choices=[
            "animals",
            "automobiles",
            "fruits",
            "furniture",
            "various",
            "vegetables",
        ],
        help="Similarity judgments of the dataset from Peterson et al. (2016) were collected for specific categories",
    )
    aa(
        "--model_names",
        type=str,
        nargs="+",
        help="models for which we want to extract featues",
    )
    aa(
        "--source",
        type=str,
        default="custom",
        choices=[
            "custom",
            "timm",
            "torchvision",
            "vissl",
            "ssl",
        ],
        help="Source of (pretrained) models",
    )
    aa(
        "--batch_size",
        metavar="B",
        type=int,
        default=64,
        help="number of images sampled during each step (i.e., mini-batch size)",
    )
    aa(
        "--features_root",
        type=str,
        default="./features",
        help="path/to/output/features",
    )
    aa(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="whether feature extraction should be performed on CPU or GPU (i.e., CUDA).",
    )
    aa(
        "--extract_cls_token",
        action="store_true",
        help="only extract [CLS] token from a ViT model",
    )
    args = parser.parse_args()
    return args


def load_extractor(
    model_name: str, source: str, device: str, extract_cls_token: bool = False
):
    if model_name.startswith("OpenCLIP"):
        if "laion" in model_name:
            meta_vars = model_name.split("_")
            name = meta_vars[0]
            variant = meta_vars[1]
            data = "_".join(meta_vars[2:])
        else:
            name, variant, data = model_name.split("_")
        model_params = dict(variant=variant, dataset=data)
    elif model_name.startswith("clip"):
        name, variant = model_name.split("_")
        model_params = dict(variant=variant)
    elif model_name.startswith("DreamSim"):
        model_name = model_name.split("_")
        name = model_name[0]
        variant = "_".join(model_name[1:])
        model_params = dict(variant=variant)
    elif extract_cls_token:
        name = model_name
        model_params = dict(extract_cls_token=True)
    else:
        name = model_name
        model_params = None

    extractor = get_extractor(
        model_name=name,
        source=source,
        device=device,
        pretrained=True,
        model_parameters=model_params,
    )
    return extractor


def feature_extraction(
    datasets: List[str],
    model_names: List[str],
    source: str,
    device: str,
    batch_size: int,
    data_root: str,
    features_root: str,
    category: str = None,
    stimulus_set: str = None,
    extract_cls_token: bool = False,
) -> None:
    for dataset in tqdm(datasets, desc="Dataset"):
        for model_name in tqdm(model_names, desc="Model"):
            extractor = load_extractor(
                model_name=model_name,
                source=source,
                device=device,
                extract_cls_token=extract_cls_token,
            )
            transformations = extractor.get_transformations()
            if dataset == "peterson":
                assert isinstance(
                    category, str
                ), "\nCategory needs to be provided for the Peterson et al. (2016;2018) dataset.\n"
                transformations = Compose(
                    [Lambda(lambda img: img.convert("RGB")), transformations]
                )
            data = load_dataset(
                name=dataset,
                data_dir=os.path.join(data_root, dataset),
                stimulus_set=stimulus_set if dataset == "free-arrangement" else None,
                category=category if dataset == "peterson" else None,
                transform=transformations,
            )
            batches = DataLoader(
                dataset=data,
                batch_size=batch_size,
                backend=extractor.get_backend(),
            )
            features = extractor.extract_features(
                batches=batches,
                module_name="visual",
                flatten_acts=True,
            )
            save_features(
                features,
                out_path=os.path.join(
                    features_root, dataset, source, model_name, "penultimate"
                ),
                file_format="hdf5",
            )


if __name__ == "__main__":
    args = parseargs()
    feature_extraction(
        datasets=args.datasets,
        model_names=args.model_names,
        source=args.source,
        device=args.device,
        batch_size=args.batch_size,
        data_root=args.data_root,
        features_root=args.features_root,
        category=args.category,
        stimulus_set=args.stimulus_set,
        extract_cls_token=args.extract_cls_token,
    )
