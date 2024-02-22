import os
import argparse
import torch
import numpy as np
import pickle as pkl
from torch.utils.data import DataLoader

from main_fewshot import create_config_dicts
from utils.probing.helpers import model_name_to_thingsvision
from thingsvision import get_extractor
from torchvision.datasets import DTD, SUN397


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_dict_path", default="/home/space/datasets/things/model_dict_all.json")
    aa("--data_root", default="/home/space/datasets/sun/")
    aa("--out_root", default="/home/space/aligned/")
    aa("--dataset", default="SUN397", choices=["SUN397", "DTD", "cifar100"])
    aa(
        "--model_names",
        type=str,
        nargs="+",
        help="models for which we want to extract featues",
    )
    aa(
        "--module",
        type=str,
        choices=["logits", "penultimate"],
        help="module for which to extract features",
    )
    aa("--overall_source", type=str, default="thingsvision")
    aa(
        "--sources",
        type=str,
        nargs="+",
        choices=[
            "custom",
            "torchvision",
            "ssl",
        ],
        help="Source of (pretrained) models",
    )
    aa(
        "--module",
        type=str,
        default="penultimate",
        help="neural network module for which to learn a linear transform",
        choices=["penultimate", "logits"],
    )
    aa("--input_dim", type=int, default=300)
    aa("--device", default="cuda")
    args = parser.parse_args()
    return args


def dataset_with_fn(cls):
    """Returns a dataset class that returns the filename as last return value."""

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, str(self._image_files[index])

    return type(
        cls.__name__,
        (cls,),
        {
            "__getitem__": __getitem__,
        },
    )


def save_embeddings(embeddings, out_path):
    if not os.path.exists(out_path):
        print("\nOutput directory does not exist...")
        print(f"Creating output directory to save results {out_path}\n")
        os.makedirs(out_path)
    with open(os.path.join(out_path, "embeddings.pkl"), "wb") as f:
        pkl.dump(embeddings, f)
        print(f"Saved embeddings to {out_path}")


def main(args):
    args.resample_testset = False
    model_cfg, data_cfg = create_config_dicts(args)
    splits = ["train", "val", "test"] if args.dataset == "DTD" else ["train", "test"]

    if args.dataset == "cifar100":
        # For cifar100, we just convert the AD features.
        embeddings_path = os.path.join(
            args.out_root, "canonical", args.module,
        )
        embeddings = {"embeddings": {}}

        # As we want to save everything into one file, we use all embeddings we can find, regardless of the model name.
        available_files = [
            file
            for file in os.listdir(args.data_root)
            if file.endswith(".npz")
            and "coarse" not in file
            and "shift" not in file
            and file.startswith("cifar100")
        ]
        print(f"Found {len(available_files)} files to convert at {args.data_root}")
        for file in available_files:
            # Try loading the embedding file
            path = os.path.join(args.data_root, file)
            try:
                ad_embeddings = np.load(path)
            except FileNotFoundError:
                print(
                    f"ERROR: File {path} not found. Make sure you the data has been extracted there via the AD scripts."
                )
                continue

            # Infer model name from file name
            model_name = "_".join(file.replace("__", "_").split("_")[1:-1]).replace(
                "p_ViT-L-14", "p_ViT-L/14"
            )
            print("Adding embeddings for", model_name)

            # Convert the AD features to a dictionary
            embeddings["embeddings"][model_name] = {
                k: v for k, v in ad_embeddings.items()
            }
        if len(embeddings["embeddings"]) > 0:
            save_embeddings(embeddings, embeddings_path)
    else:
        for model_name, module, source in zip(
            args.model_names, model_cfg.modules, args.sources
        ):
            print(
                f"#################### Extracting {args.dataset} features for {model_name}({source})"
            )

            name, model_params = model_name_to_thingsvision(model_name)

            # For SUN397 and DTD, we extract the features from the models
            embeddings_path = os.path.join(args.out_root, f"{args.dataset}/embeddings")

            extractor = get_extractor(
                model_name=name,
                source=source,
                device=args.device,
                pretrained=True,
                model_parameters=model_params,
            )

            embeddings = {}
            for split in splits:
                print(f"Split: {split}")
                if args.dataset == "DTD":
                    dataset = dataset_with_fn(DTD)(
                        root=data_cfg.root,
                        split=split,
                        download=False,
                        transform=extractor.get_transformations(),
                    )
                elif args.dataset == "SUN397":
                    dataset = dataset_with_fn(SUN397)(
                        root=data_cfg.root,
                        download=False,
                        transform=extractor.get_transformations(),
                    )
                    if split == "train":
                        split_file = "Training_01.txt"
                    elif split == "test":
                        split_file = "Testing_01.txt"
                    with open(os.path.join(dataset.root, split_file)) as f:
                        lines = f.read()
                    file_names = [l for l in lines.split("\n") if not l == ""]
                    dataset._image_files = [
                        os.path.join(dataset._data_dir, fn[1:]) for fn in file_names
                    ]
                    dataset._labels = [
                        dataset.class_to_idx["/".join(path.split("/")[2:-1])]
                        for path in file_names
                    ]
                else:
                    raise ValueError(f"Unknown dataset {args.dataset}")
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=64, shuffle=False, drop_last=False
                )
                for x, y, fns in loader:
                    features = extractor.extract_features(
                        batches=[x],
                        module_name=module,
                        flatten_acts=True,
                    )
                    for fn, feat in zip(fns, features):
                        embeddings[fn] = feat

            out_path = os.path.join(embeddings_path, source, model_name, args.module)
            save_embeddings(embeddings, out_path)


if __name__ == "__main__":
    args = parseargs()
    main(args)
