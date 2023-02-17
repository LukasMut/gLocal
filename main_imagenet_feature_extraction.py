import argparse
import os
from typing import Any, Dict

import torch
from thingsvision import get_extractor
from thingsvision.utils.data import ImageDataset
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
    return model_cfg


def load_extractor(model_cfg: Dict[str, str]) -> Any:
    """Load extractor for specific model and source."""
    model_name = model_cfg["model"]
    if model_name.startswith("OpenCLIP"):
        name, variant, data = model_name.split("_")
        model_params = dict(variant=variant, dataset=data)
    elif model_name.startswith("clip"):
        name, variant = model_name.split("_")
        model_params = dict(variant=variant)
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


def save_features(features: Tensor, out_path: str, split: str) -> None:
    """Save ImageNet features as single PyTorch tensors to disk."""
    split_path = os.path.join(out_path, split)
    if not os.path.exists(out_path):
        print("\nCreating output directory for saving ImageNet features...\n")
        os.makedirs(out_path, exist_ok=True)
    for i, x in tqdm(enumerate(features, start=1), desc="Features"):
        torch.save(x, os.path.join(split_path, f"imagenet_features_{i:07d}.pt"))


def extract(
    imagenet_root: str,
    model_cfg: Dict[str, str],
    batch_size: int,
    num_workers: int,
    out_path: str,
    resize_dim: 256,
    crop_dim: 224,
) -> None:
    """Run extraction pipeline."""
    extractor = load_extractor(model_cfg)
    imagenet_train_set = ImageDataset(
        os.path.join(imagenet_root, "train_set"),
        out_path=os.path.join(out_path, "train"),
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(
            resize_dim=resize_dim, crop_dim=crop_dim
        ),
    )
    imagenet_val_set = ImageDataset(
        os.path.join(imagenet_root, "val_set"),
        out_path=os.path.join(out_path, "val"),
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(
            resize_dim=resize_dim, crop_dim=crop_dim
        ),
    )
    train_batches = DataLoader(
        dataset=imagenet_train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    val_batches = DataLoader(
        dataset=imagenet_val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    train_features = extractor.extract_features(
        batches=train_batches,
        module_name=model_cfg["module"],
        flatten_acts=True,
        output_type="tensor",
    )
    save_features(train_features, out_path=out_path, split="train")
    del train_features
    val_features = extractor.extract_features(
        batches=val_batches,
        module_name=model_cfg["module"],
        flatten_acts=True,
        output_type="tensor",
    )
    save_features(val_features, out_path=out_path, split="val")
    del val_features


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    model_cfg = create_model_config(args)
    out_path = os.path.join(args.out_path, model_cfg["source"], model_cfg["model"], args.module)
    if not os.path.exists(out_path):
        print("\nCreating output directory for saving ImageNet features...\n")
        os.makedirs(out_path, exist_ok=True)
    extract(
        imagenet_root=args.imagenet_root,
        model_cfg=model_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_path=out_path,
    )
