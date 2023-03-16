import argparse
import itertools
import os
import pickle
import warnings
from datetime import datetime
from typing import Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from ml_collections import config_dict
from thingsvision import get_extractor

import utils
from downstream.fewshot.data import load_dataset
from downstream.fewshot.predictors import train_regression, train_knn, test_regression
from downstream.fewshot.utils import is_embedding_source, apply_transform
from main_model_sim_eval import get_module_names
from utils.probing.helpers import model_name_to_thingsvision

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/things")
    aa("--dataset", type=str, help="Which dataset to use", default="things")
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
            "google",
            "loss",
            "imagenet",
            "torchvision",
            "vit_same",
            "vit_best",
        ],
        help="Source of (pretrained) models",
    )
    aa(
        "--model_dict_path",
        type=str,
        default="/home/space/datasets/things/model_dict.json",
        help="Path to the model_dict.json",
    )
    aa(
        "--n_shot",
        type=int,
        nargs="+",
        help="Number samples per class for training",
        default=10,
    )
    aa(
        "--n_test",
        type=int,
        help="Number samples per class for testing",
        default=100,
    )
    aa(
        "--n_reps",
        type=int,
        help="Number of repetitions per experiment",
        default=1,
    )
    aa(
        "--input_dim",
        type=int,
        help="Side-length of the input images.",
        default=32,
    )
    aa(
        "--regressor_type",
        type=str,
        nargs="+",
        choices=["ridge", "knn"],
        help="Few shot model.",
    )
    aa(
        "--n_classes",
        type=int,
        help="Number of classes",
    )
    aa("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    aa(
        "--things_embeddings_path",
        type=str,
        default="/home/space/datasets/things/embeddings/model_features_per_source.pkl",
        help="path/to/things/embeddings/file",
    )
    aa(
        "--transforms_root",
        type=str,
        default="/home/space/datasets/things",
        help="path/to/embeddings",
    )
    aa(
        "--transform_type",
        type=str,
        default="without_norm",
        choices=["without_norm", "with_norm"],
        help="type of transformation matrix being used",
    )
    aa("--out_dir", type=str, help="directory to save the results to")
    aa("--rnd_seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()
    return args


def get_features_targets(
    class_ids,
    model_name,
    model_params,
    source,
    module,
    module_type,
    data_cfg,
    batch_size,
    train,
    ids_subset=None,
    n_batches=1,
    shuffle=False,
    device: str = "cpu",
    embeddings: Optional[Array] = None,
):
    ids_subset = class_ids if ids_subset is None else ids_subset
    dataset_is_embedded = is_embedding_source(source)

    if dataset_is_embedded:
        # Load the dataset from an embedding source
        dataset = load_dataset(
            name=data_cfg.name,
            data_dir=data_cfg.root,
            train=train,
            embeddings=embeddings,
        )
    else:
        complete_model_name = model_name + ("" if model_params is None else ("_" + model_params["variant"]))
        try:
            # Try to load the embeddings from disk
            embeddings_path = os.path.join(data_cfg.embeddings_root, source, complete_model_name, module_type)
            with open(os.path.join(embeddings_path, "embeddings.pkl"), "rb") as f:
                embeddigns = pickle.load(f)
            dataset = load_dataset(
                    name=data_cfg.name,
                    data_dir=data_cfg.root,
                    train=train,
                    embeddings=embeddigns,
            )
            dataset_is_embedded = True
        except (FileNotFoundError, TypeError):
            # If the embeddings are not found or embeddings_rood is None, extract embeddings
            extractor = get_extractor(
                model_name=model_name,
                source=source,
                device=device,
                pretrained=True,
                model_parameters=model_params,
            )
            dataset = load_dataset(
                name=data_cfg.name,
                data_dir=data_cfg.root,
                train=train,
                transform=extractor.get_transformations(),
            )
    features_all = []
    Y_all = []
    for i_batch in range(n_batches):
        X = None
        Y = None
        for i_cls_id, cls_id in enumerate(class_ids):
            if cls_id not in ids_subset:
                continue
            try:
                subset_indices = [
                    i_cls for i_cls, cls in enumerate(dataset.targets) if cls == cls_id
                ]
            except AttributeError:
                subset_indices = [
                    i_cls for i_cls, cls in enumerate(dataset._labels) if cls == cls_id
                ]
            subset = torch.utils.data.Subset(
                dataset,
                subset_indices,
            )
            batches = torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=2,
                worker_init_fn=lambda id: np.random.seed(id + i_batch * 4),
            )

            for x, y in batches:
                if X is None:
                    X = [x.to(device)]
                    Y = [i_cls_id] * len(y)
                else:
                    X += [x.to(device)]
                    Y += [i_cls_id] * len(y)
                break
        Y = np.array(Y)

        if dataset_is_embedded:
            features = torch.stack(list(itertools.chain.from_iterable(X)), dim=0).detach().cpu().numpy()
        elif (
                    source == "torchvision"
                    and module in ["penultimate", "encoder.ln"]
                    and model_name.startswith("vit")
            ):
            features = extractor.extract_features(
                    batches=X,
                    module_name=module,
                    flatten_acts=False,
            )
            features = features[:, 0].clone()  # select classifier token
            features = features.reshape((features.shape[0], -1))
        else:
            features = extractor.extract_features(
                batches=X,
                module_name=module,
                flatten_acts=True,
            )

        features_all.append(features)
        Y_all.append(Y)

    return features_all, Y_all


def create_config_dicts(args, embedding_keys=None) -> Tuple[FrozenDict, FrozenDict]:
    """Create data and model config dictionaries."""
    model_config = utils.evaluation.load_model_config(args.model_dict_path)
    model_cfg = config_dict.ConfigDict()
    data_cfg = config_dict.ConfigDict()
    model_cfg.module_type = args.module
    model_cfg.sources = args.sources
    model_cfg.input_dim = args.input_dim
    if embedding_keys is not None:
        model_cfg.embeddings_root = args.embeddings_root.split("/")[-1]
        model_cfg.names = embedding_keys
    else:
        if hasattr(args, "embeddings_root"):
            embeddings_root = args.embeddings_root
        else:
            embeddings_root = None
        model_cfg.embeddings_root = embeddings_root
        data_cfg.embeddings_root = embeddings_root
        model_cfg.names = args.model_names
    model_cfg.modules = get_module_names(model_config, model_cfg.names, args.module)
    model_cfg = config_dict.FrozenConfigDict(model_cfg)
    data_cfg.root = args.data_root
    data_cfg.name = args.dataset
    try:
        data_cfg.category = args.category
    except:
        data_cfg.category = None
    data_cfg = config_dict.FrozenConfigDict(data_cfg)

    return model_cfg, data_cfg


def get_regressor(train_features: Array, train_targets: Array, regressor_type: str, k: Optional[int] = None):
    if regressor_type == "ridge":
        regressor = train_regression(
                train_targets, train_features, k=k
        )
    elif regressor_type == "knn":
        regressor = train_knn(train_targets, train_features)
    else:
        raise ValueError(f"Unknown regressor: {regressor_type}")
    return regressor


def run(
    n_shot: int,
    n_test: int,
    n_reps: int,
    class_id_sets: List,
    device: str,
    model_cfg: FrozenDict,
    data_cfg: FrozenDict,
    features_things: Array,
    transforms: Array,
    transform_type: str = None,
    regressor_type: str = "ridge",
):
    transform_options = [False, True]

    results = []
    for class_id_set in class_id_sets:
        for model_name, module, source in zip(
            model_cfg.names, model_cfg.modules, model_cfg.sources
        ):
            # Resolve family name
            name, model_params = model_name_to_thingsvision(model_name)
            family_name = utils.analyses.get_family_name(model_name)
            module_type = model_cfg.module_type

            # Extract train features
            train_features_original_all, train_targets_all = get_features_targets(
                class_id_set,
                name,
                model_params,
                source,
                module,
                module_type,
                data_cfg,
                n_shot,
                train=True,
                n_batches=n_reps,
                shuffle=True,
                device=device,
            )

            things_mean = np.mean(
                features_things[source][model_name][model_cfg.module_type],
                # axis=0,
            )
            things_std = np.std(
                features_things[source][model_name][model_cfg.module_type],
                # axis=0,
            )

            # Train regression w and w/o transform
            regressors = {to: [] for to in transform_options}
            for (train_features_original, train_targets) in zip(
                train_features_original_all, train_targets_all
            ):
                for use_transforms in transform_options:
                    if use_transforms:
                        try:
                            transform = transforms[source][model_name][
                                model_cfg.module_type
                            ]
                        except KeyError:
                            warnings.warn(
                                message=f"\nCould not find transformation matrix for {model_name}.\nSkipping evaluation for {model_name} and continuing with next model...\n",
                                category=UserWarning,
                            )
                            continue
                        train_features = apply_transform(
                            train_features_original,
                            transform,
                            things_mean,
                            things_std,
                            transform_type=transform_type,
                        )
                    else:
                        train_features = train_features_original - things_mean

                    regressor = get_regressor(train_features, train_targets, regressor_type, n_shot)
                    regressors[use_transforms].append(regressor)

            # Extract and evaluate features w and w/o transform. Due to memory constraints, for each class individually.
            for i_rep in range(n_reps):
                accuracies = {a: [] for a in transform_options}
                for cls_id in [class_id_set]:
                    test_features_original, test_targets = get_features_targets(
                        class_id_set,
                        name,
                        model_params,
                        source,
                        module,
                        module_type,
                        data_cfg,
                        n_test,
                        train=False,
                        shuffle=False,
                        ids_subset=cls_id if type(cls_id) == list else [cls_id],
                        device=device,
                    )
                    test_features_original = test_features_original[0]
                    test_targets = test_targets[0]

                    for use_transforms in transform_options:
                        if use_transforms:
                            try:
                                transform = transforms[source][model_name][
                                    model_cfg.module_type
                                ]
                            except KeyError:
                                warnings.warn(
                                    message=f"\nCould not find transformation matrix for {model_name}.\nSkipping evaluation for {model_name} and continuing with next model...\n",
                                    category=UserWarning,
                                )
                                continue
                            test_features = apply_transform(
                                test_features_original,
                                transform,
                                things_mean,
                                things_std,
                                transform_type=transform_type,
                            )
                        else:
                            test_features = test_features_original - things_mean

                        acc, pred = test_regression(
                            regressors[use_transforms][i_rep],
                            test_targets,
                            test_features,
                        )
                        accuracies[use_transforms].append(acc)

                # Store results for all classes
                for use_transforms in transform_options:
                    summary = {
                        "accuracy": np.mean(accuracies[use_transforms]),
                        "model": model_name,
                        "module": model_cfg.module_type,
                        "source": source,
                        "family": family_name,
                        "dataset": data_cfg.name,
                        "transform": use_transforms,
                        "classes": class_id_set,
                        "n_train": n_shot,
                        "repetition": i_rep,
                        "transform_type": transform_type if use_transforms else None,
                        "regressor": regressor_type,
                    }
                    results.append(summary)

            print(summary)

    results = pd.DataFrame(results)
    return results


def _add_model(model_cfg, f):
    return f(model_cfg=model_cfg)


if __name__ == "__main__":
    start_t = datetime.now()

    # parse arguments
    args = parseargs()
    class_id_sets = [[i for i in range(args.n_classes)]]
    n_shot = args.n_shot
    n_test = args.n_test
    device = torch.device(args.device)

    if args.embeddings_root is not None:
        embeddings = utils.evaluation.load_embeddings(
            embeddings_root=args.embeddings_root,
            module="embeddings" if args.module == "penultimate" else "logits",
        )
    else:
        embeddings = None
    model_cfg, data_cfg = create_config_dicts(args, embeddings.keys())

    # Load transforms and things embeddings
    features_things = utils.evaluation.load_features(path=args.things_embeddings_path)
    transforms = utils.evaluation.helpers.load_transforms(
        root=args.transforms_root, type=args.transform_type
    )

    # Reduce to the needed models
    transforms = {
        src: {
            mdl: transforms[src][mdl]
            for mdl_i, mdl in enumerate(args.model_names)
            if args.sources[mdl_i] == src
        }
        for src in args.sources
    }

    features_things = {
        src: {
            mdl: features_things[src][mdl]
            for mdl_i, mdl in enumerate(args.model_names)
            if args.sources[mdl_i] == src
        }
        for src in args.sources
    }

    # Do few-shot
    all_results = []
    regressor_types = args.regressor_type
    n_shots = args.n_shot
    for regressor_type in regressor_types:
        for shots in n_shots:
            if regressor_type == "ridge" and shots == 1:
                continue
            args.n_shot = shots
            args.regressor_type = regressor_type
            model_cfg, data_cfg = create_config_dicts(args, embeddings.keys())

            results = run(
                n_shot=args.n_shot,
                n_test=args.n_test,
                n_reps=args.n_reps,
                class_id_sets=class_id_sets,
                device=args.device,
                model_cfg=model_cfg,
                data_cfg=data_cfg,
                features_things=features_things,
                transforms=transforms,
                transform_type=args.transform_type,
                regressor_type=args.regressor_type,
            )
            all_results.append(results)
    results = pd.concat(all_results)

    out_path = os.path.join(
        args.out_dir, args.dataset, args.overall_source, args.module
    )
    if not os.path.exists(out_path):
        print("\nOutput directory does not exist...")
        print("Creating output directory to save results...\n")
        os.makedirs(out_path)

    results.to_pickle(os.path.join(out_path, "fewshot_results.pkl"))

    print("Elapsed time (init):", datetime.now() - start_t)
