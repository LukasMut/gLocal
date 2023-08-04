import argparse
import os
import re
from typing import Any, Dict, List

import torch
from thingsvision import get_extractor
from thingsvision.utils.data import ImageDataset
from thingsvision.utils.storing import save_features
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils

Tensor = torch.Tensor


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--imagenet_root",
        type=str,
        help="path/to/imagenet/data/folder",
        default="/home/space/datasets/imagenet/2012/",
    )
    aa("--out_path", type=str, help="path/to/imagenet/output/features")
    aa(
        "--model_dict_path",
        type=str,
        default="/home/space/datasets/things/model_dict_all.json",
        help="Path to the model_dict.json",
    )
    aa("--model", type=str)
    aa(
        "--module",
        type=str,
        default="penultimate",
        help="neural network module for which to learn a linear transform",
        choices=["penultimate", "logits"],
    )
    aa(
        "--source",
        type=str,
        default="custom",
        choices=[
            "custom",
            "ssl",
            "torchvision",
        ],
    )
    aa(
        "--batch_size",
        type=int,
        default=512,
        help="Use a power of 2 for running extraction process on GPU",
        choices=[64, 128, 256, 512, 1024, 2048],
    )
    aa(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers used for loading data",
        choices=[4, 8, 10, 12, 16, 20, 32],
    )
    aa(
        "--out_format",
        type=str,
        default="hdf5",
        help="With which data type ImageNet feature matrices should be saved to disk",
        choices=["hdf5", "pt"],
    )
    aa(
        "--splits",
        type=str,
        default=["train", "val"],
        nargs="+",
        help="Which splits to extract features for",
        choices=[
            "train",
            "val",
        ],
    )
    aa(
        "--extract_cls_token",
        action="store_true",
        help="whether to exclusively extract the [cls] token for DINO models",
    )
    aa("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    return args


def create_model_config(args) -> Dict[str, str]:
    """Create config dict for model hyperparameters."""
    model_cfg = {}
    model_config = utils.evaluation.load_model_config(args.model_dict_path)
    model_cfg["model"] = args.model
    model_cfg["module"] = model_config[args.model][args.module]["module_name"]
    model_cfg["source"] = args.source
    model_cfg["device"] = args.device
    model_cfg["extract_cls_token"] = args.extract_cls_token
    return model_cfg


def load_extractor(model_cfg: Dict[str, str]) -> Any:
    """Load extractor for specific model and source."""
    model_name = model_cfg["model"]
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
    elif model_cfg["extract_cls_token"]:
        name = model_name
        model_params = dict(extract_cls_token=True)
    else:
        name = model_name
        model_params = None
    extractor = get_extractor(
        model_name=name,
        source=model_cfg["source"],
        device=model_cfg["device"],
        pretrained=True,
        model_parameters=model_params,
    )
    return extractor


def save_features_sequentially(features: Tensor, out_path: str, split: str) -> None:
    """Save ImageNet features as single PyTorch tensors to disk."""
    split_path = os.path.join(out_path, split)
    if not os.path.exists(split_path):
        print("\nCreating output directory for saving ImageNet features...\n")
        os.makedirs(split_path, exist_ok=True)
    for i, x in tqdm(enumerate(features, start=1), desc="Features"):
        torch.save(x.clone(), os.path.join(split_path, f"imagenet_features_{i:07d}.pt"))


def extract(
    imagenet_root: str,
    model_cfg: Dict[str, str],
    batch_size: int,
    num_workers: int,
    out_path: str,
    splits: List[str],
    out_format: str,
    resize_dim: int = 256,
    crop_dim: int = 224,
) -> None:
    """Run extraction pipeline."""
    extractor = load_extractor(model_cfg)

    for split in splits:
        imagenet_split_set = ImageDataset(
            os.path.join(imagenet_root, "_".join((split, "set"))),
            out_path=os.path.join(out_path, split),
            backend=extractor.get_backend(),
            transforms=extractor.get_transformations(
                resize_dim=resize_dim, crop_dim=crop_dim
            ),
        )
        batches = DataLoader(
            dataset=imagenet_split_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )
        if (
            (model_cfg["source"] == "torchvision" or model_cfg["source"] == "ssl")
            and model_cfg["module"] == "penultimate"
            and re.search(r"vit", model_cfg["model"])
        ):
            features = extractor.extract_features(
                batches=batches,
                module_name=model_cfg["module"],
                flatten_acts=False,
                output_type="tensor",
            )
            features = features[:, 0, :].copy()
        else:
            features = extractor.extract_features(
                batches=batches,
                module_name=model_cfg["module"],
                flatten_acts=True,
                output_type="tensor",
            )
        if out_format == "pt":
            save_features_sequentially(features, out_path=out_path, split=split)
        elif out_format == "hdf5":
            save_features(
                features.cpu().numpy(),
                out_path=os.path.join(out_path, split),
                file_format=out_format,
            )
        else:
            raise ValueError(
                "\nData type for saving features to disk must be set to either 'pt' or 'hdf5'.\n"
            )
        del features


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    model_cfg = create_model_config(args)
    out_path = os.path.join(
        args.out_path, model_cfg["source"], model_cfg["model"], args.module
    )
    if not os.path.exists(out_path):
        print("\nCreating output directory for saving ImageNet features...\n")
        os.makedirs(out_path, exist_ok=True)
    extract(
        imagenet_root=args.imagenet_root,
        model_cfg=model_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_path=out_path,
        splits=args.splits,
        out_format=args.out_format,
    )
