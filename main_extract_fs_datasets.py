import torch
import argparse
from torch.utils.data import DataLoader
import os
import pickle as pkl
from main_fewshot import create_config_dicts
from thingsvision import get_extractor
from torchvision.datasets import CIFAR100, DTD, SUN397


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_dict_path", default="/home/space/datasets/things/model_dict_all.json")
    aa("--data_root", default="/home/space/datasets/sun/")  # DTD/")#
    aa("--out_root", default="/home/space/aligned/")
    aa("--input_dim", type=int, default=300)
    aa("--dataset", default="SUN397")  # "DTD")#
    aa("--device", default="cuda")
    args = parser.parse_args()
    return args


def dataset_with_fn(cls):
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


def main():
    # parse arguments
    args = parseargs()

    class Arguments:
        data_root = args.data_root
        out_root = args.out_root
        dataset = args.dataset
        model_names = []
        sources = []
        module = "penultimate"
        model_dict_path = args.model_dict_path
        device = "cuda"
        input_dim = args.input_dim
        splits = (
            ["train", "val", "test"] if args.dataset == "DTD" else ["train", "test"]
        )

    args = Arguments()

    args.model_names = [
        "dino-vit-base-p8",
        "dino-vit-base-p16",
        "dinov2-vit-base-p14",
        "dinov2-vit-large-p14",
        "OpenCLIP_ViT-L-14_laion400m_e32",
        "OpenCLIP_ViT-L-14_laion2b_s32b_b82k",
        "alexnet",
        "vgg16",
    ]
    args.sources = [
        "ssl",
        "ssl",
        "ssl",
        "ssl",
        "custom",
        "custom",
        "torchvision",
        "torchvision",
    ]
    args.resample_testset = False

    model_cfg, data_cfg = create_config_dicts(args)
    embeddings_path = os.path.join(args.out_root, f"{args.dataset}/embeddings")

    for model_name, module, source in zip(
        args.model_names, model_cfg.modules, args.sources
    ):
        print("####################", model_name, source)

        if model_name.startswith("OpenCLIP"):
            split = model_name.split("_")
            name = split[0]
            variant = split[1]
            data = "_".join(split[2:])
            model_params = dict(variant=variant, dataset=data)
        elif model_name.startswith("clip"):
            name, variant = model_name.split("_")
            model_params = dict(variant=variant)
        else:
            name = model_name
            model_params = None
            if name.startswith("dino"):
                model_params = dict(extract_cls_token=True)

        extractor = get_extractor(
            model_name=name,
            source=source,
            device=args.device,
            pretrained=True,
            model_parameters=model_params,
        )

        embeddings = {}
        for split in args.splits:
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
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=False, drop_last=False  # 1024,
            )
            for x, y, fns in loader:
                if (
                    source == "torchvision"
                    and module in ["penultimate", "encoder.ln"]
                    and model_name.startswith("vit")
                ):
                    features = extractor.extract_features(
                        batches=[x],
                        module_name=module,
                        flatten_acts=False,
                    )
                    features = features[:, 0].clone()  # select classifier token
                    features = features.reshape((features.shape[0], -1))
                else:
                    features = extractor.extract_features(
                        batches=[x],
                        module_name=module,
                        flatten_acts=True,
                    )
                for fn, feat in zip(fns, features):
                    embeddings[fn] = feat

        out_path = os.path.join(embeddings_path, source, model_name, args.module)
        if not os.path.exists(out_path):
            print("\nOutput directory does not exist...")
            print(f"Creating output directory to save results {out_path}\n")
            os.makedirs(out_path)
        with open(os.path.join(out_path, "embeddings.pkl"), "wb") as f:
            pkl.dump(embeddings, f)


if __name__ == "__main__":
    main()
