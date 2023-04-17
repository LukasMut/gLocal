import argparse
import itertools
import os
import pickle
from datetime import datetime
from typing import Any, List, Tuple, Optional, Dict, Union

import numpy as np
import pandas as pd
import torch
from ml_collections import config_dict
from thingsvision import get_extractor

import utils
from downstream.fewshot.breeds_sets import get_breeds_task
from downstream.fewshot.cifar import get_cifar100_coarse_map
from downstream.fewshot.data import load_dataset
from downstream.fewshot.predictors import test_regression, get_regressor
from downstream.fewshot.utils import is_embedding_source
from main_model_sim_eval import get_module_names
from main_glocal_probing_efficient import get_combination
from utils.probing.helpers import model_name_to_thingsvision
from utils.evaluation.transforms import GlobalTransform, GlocalTransform

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any

BREEDS_TASKS = ("living17", "entity13", "entity30", "nonliving26")


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    # Base arguments
    aa("--data_root", type=str, help="path/to/things")
    aa("--dataset", type=str, help="Which dataset to use", default="imagenet")
    aa(
        "--task",
        type=str,
        choices=["none", "coarse"] + list(BREEDS_TASKS),
        help="Which task to do",
        default="none",
    )
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
        "--input_dim",
        type=int,
        help="Side-length of the input images.",
        default=32,
    )
    # Few shot arguments
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
    aa(
        "--class_id_set",
        type=int,
        nargs="+",
        help="Classes to use",
        default=None,
    )
    aa(
        "--resample_testset",
        action="store_true",
        help="Whether to re-sample the test samples for each repetition. Should be True if not all test samples are to be used in each iter",
    )
    aa(
        "--sample_per_superclass",
        action="store_true",
        help="Whether to sample the shots for each superclass, rather than each class.",
    )
    # Transform arguments
    aa("--optim", type=str, default="SGD", choices=["Adam", "AdamW", "SGD"])
    aa(
        "--etas",
        type=float,
        default=1e-3,
        nargs="+",
    )
    aa(
        "--lmbdas",
        type=float,
        default=1e-3,
        nargs="+",
        help="Relative contribution of the l2 or identity regularization penality",
    )
    aa(
        "--alphas",
        type=float,
        default=1e-1,
        nargs="+",
        help="Relative contribution of the contrastive loss term",
    )
    aa(
        "--taus",
        type=float,
        default=1,
        nargs="+",
        help="temperature value for contrastive learning objective",
    )
    aa(
        "--contrastive_batch_sizes",
        type=int,
        default=1024,
        nargs="+",
        metavar="B_C",
    )
    aa(
        "--glob",
        action="store_true",
        help="Whether to load global probes.",
    )
    # Misc arguments
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
        default="/home/space/datasets/things/probing/results",
        help="path/to/embeddings",
    )
    aa(
        "--embeddings_root",
        type=str,
        default=None,
        help="path/to/embeddings of the dataset",
    )
    aa("--out_dir", type=str, help="directory to save the results to")
    aa("--rnd_seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()
    return args


def get_subset_indices(dataset, cls_id: Union[int, List[int]]):
    if type(cls_id) == int:
        cls_id = [cls_id]
    try:
        subset_indices = [
            i_cls for i_cls, cls in enumerate(dataset.targets) if cls in cls_id
        ]
    except AttributeError:
        subset_indices = [
            i_cls for i_cls, cls in enumerate(dataset._labels) if cls in cls_id
        ]
    return subset_indices

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
    n_batches=1,  # number of reps
    shuffle=False,
    device: str = "cpu",
    embeddings: Optional[Array] = None,
    superclass_mapping: Optional[Dict] = None,
    sample_per_superclass: bool = False,
):
    ids_subset = class_ids if ids_subset is None else ids_subset
    dataset_is_embedded = is_embedding_source(source) or embeddings is not None

    if dataset_is_embedded:
        # Load the dataset from an embedding source
        dataset = load_dataset(
            name=data_cfg.name,
            data_dir=data_cfg.root,
            train=train,
            embeddings=embeddings,
        )
    else:
        complete_model_name = model_name + (
            "" if model_params is None else ("_" + model_params["variant"])
        )
        try:
            # Try to load the embeddings from disk
            embeddings_path = os.path.join(
                data_cfg.embeddings_root, source, complete_model_name, module_type
            )
            if data_cfg.name not in ["imagenet"]:
                # For all other datasets, we can load the embeddings from a single file
                with open(os.path.join(embeddings_path, "embeddings.pkl"), "rb") as f:
                    embeddings = pickle.load(f)
            else:
                # For imagenet, we need to load the embeddings from individual files
                embeddings = None
            dataset = load_dataset(
                name=data_cfg.name,
                data_dir=data_cfg.root,
                train=train,
                embeddings=embeddings,
                embeddings_root=embeddings_path,
            )
            dataset_is_embedded = True
        except (FileNotFoundError, TypeError):
            # If the embeddings are not found or embeddings_root is None, extract embeddings
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

    if sample_per_superclass:
        # sampe one barch or size #shots per superclass, rather than one batch per class
        n_superclasses = len(set(superclass_mapping.values()))
        class_ids = [
            [ci for ci in class_ids if superclass_mapping[ci] == i]
            for i in range(n_superclasses)
        ]
    features_all = []
    Y_all = []
    for i_batch in range(n_batches):
        X = None
        Y = None
        indices = []
        for i_cls_id, cls_id in enumerate(class_ids):
            if type(cls_id) == int and cls_id not in ids_subset:
                continue
            subset_indices = get_subset_indices(dataset, cls_id)
            indices += list(np.random.choice(subset_indices, size=batch_size, replace=False))

        subset = torch.utils.data.Subset(
            dataset,
            indices,
        )
        batches = torch.utils.data.DataLoader(
            subset,
            batch_size=len(indices),
            shuffle=shuffle,
            num_workers=4,
            worker_init_fn=lambda id: np.random.seed(id + i_batch * 4),
        )

        for x, y in batches:
            X = [x.to(device)]
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = torch.argmax(y, dim=1)
            if superclass_mapping is not None:
                y = [superclass_mapping[int(y_elem)] for y_elem in y]
            Y = np.array(y)
            break

        if dataset_is_embedded:
            features = (
                torch.stack(list(itertools.chain.from_iterable(X)), dim=0)
                .detach()
                .cpu()
                .numpy()
            )
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
        model_cfg.embeddings_root = args.embeddings_root  # .split("/")[-1]
        model_cfg.names = [k for k in embedding_keys]
    else:
        if hasattr(args, "embeddings_root"):
            embeddings_root = args.embeddings_root
        else:
            embeddings_root = None
        model_cfg.embeddings_root = embeddings_root
        model_cfg.names = args.model_names
    data_cfg.embeddings_root = model_cfg.embeddings_root
    model_cfg.modules = get_module_names(model_config, model_cfg.names, args.module)
    model_cfg = config_dict.FrozenConfigDict(model_cfg)
    data_cfg.root = args.data_root
    data_cfg.name = args.dataset
    data_cfg.resample_testset = args.resample_testset
    try:
        data_cfg.category = args.category
    except:
        data_cfg.category = None
    data_cfg = config_dict.FrozenConfigDict(data_cfg)

    return model_cfg, data_cfg


def run(
    n_shot: int,
    n_test: int,
    n_reps: int,
    class_id_set: List,
    device: str,
    model_cfg: FrozenDict,
    data_cfg: FrozenDict,
    transforms: Dict,
    regressor_type: str = "ridge",
    class_id_set_test: Optional[List] = None,
    superclass_mapping: Optional[Dict] = None,
    sample_per_superclass: bool = False,
    model_id_in_cfg: int = 0,
    embeddings: Optional[Dict] = None,
):
    transform_options = [False, True]

    if class_id_set_test is None:
        class_id_set_test = class_id_set
        print("Using training classes for testing")

    model_name, module, source = (
        model_cfg.names[model_id_in_cfg],
        model_cfg.modules[model_id_in_cfg],
        model_cfg.sources[model_id_in_cfg],
    )
    # Resolve family name
    name, model_params = model_name_to_thingsvision(model_name)
    family_name = utils.analyses.get_family_name(model_name)
    module_type = model_cfg.module_type

    if embeddings is not None:
        embeddings = embeddings[model_name]

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
        superclass_mapping=superclass_mapping,
        sample_per_superclass=sample_per_superclass,
        embeddings=embeddings,
    )

    try:
        things_mean = transforms[source][model_name].things_mean
    except:
        things_mean = transforms[source][model_name].transform["mean"]

    # Train regression w and w/o transform
    regressors = {to: [] for to in transform_options}
    for (train_features_original, train_targets) in zip(
        train_features_original_all, train_targets_all
    ):
        # This loops over the repetitions
        for use_transforms in transform_options:
            if use_transforms:
                train_features = transforms[source][model_name].transform_features(
                    train_features_original
                )
            else:
                train_features = train_features_original - things_mean

            regressor = get_regressor(
                train_features, train_targets, regressor_type, n_shot
            )
            regressors[use_transforms].append(regressor)

    # Extract and evaluate features w and w/o transform. Due to memory constraints, for each class individually.
    results = []
    for i_rep in range(n_reps):
        accuracies = {a: None for a in transform_options}
        if i_rep == 0 or data_cfg.resample_testset:
            test_features_original, test_targets = get_features_targets(
                class_id_set_test,
                name,
                model_params,
                source,
                module,
                module_type,
                data_cfg,
                n_test,
                train=False,
                device=device,
                superclass_mapping=superclass_mapping,
                # sample_per_superclass=sample_per_superclass,
                embeddings=embeddings,
            )
            test_features_original = test_features_original[0]
            test_targets = test_targets[0]

        for use_transforms in transform_options:
            if use_transforms:
                test_features = transforms[source][model_name].transform_features(
                    test_features_original
                )
            else:
                test_features = test_features_original - things_mean

            acc, pred = test_regression(
                regressors[use_transforms][i_rep],
                test_targets,
                test_features,
            )
            accuracies[use_transforms] = acc

        # Store results for all classes
        for use_transforms in transform_options:
            summary = {
                "accuracy": accuracies[use_transforms],
                "model": model_name,
                "module": model_cfg.module_type,
                "source": source,
                "family": family_name,
                "dataset": data_cfg.name,
                "transform": use_transforms,
                "classes": list(set(class_id_set).union(set(class_id_set_test))),
                "n_train": n_shot,
                "repetition": i_rep,
                "regressor": regressor_type,
                "samples_per_superclass": sample_per_superclass,
            }
            for att in [
                "optim",
                "eta",
                "lmbda",
                "alpha",
                "tau",
                "contrastive_batch_size",
            ]:
                try:
                    summary[att] = transforms[source][model_name].__getattribute__(att)
                except:
                    summary[att] = None

            results.append(summary)
    print(summary)  # prints last summary - TODO: remove

    results = pd.DataFrame(results)
    return results


def _add_model(model_cfg, f):
    return f(model_cfg=model_cfg)


if __name__ == "__main__":
    start_t = datetime.now()

    # parse arguments
    args = parseargs()
    if args.task in BREEDS_TASKS:
        class_id_set, class_id_set_test, superclass_mapping = get_breeds_task(args.task)
    elif args.dataset == "cifar100" and args.task == "coarse":
        superclass_mapping = get_cifar100_coarse_map()
        class_id_set = class_id_set_test = [i for i in range(100)]
    else:
        args.task = None
        superclass_mapping = None
        if args.class_id_set is None:
            class_id_set = [i for i in range(args.n_classes)]
        else:
            class_id_set = args.class_id_set
        class_id_set_test = class_id_set
    n_shot = args.n_shot
    n_test = args.n_test
    device = torch.device(args.device)

    # Load embeddings
    if args.embeddings_root is not None:
        try:
            embeddings = utils.evaluation.load_embeddings(
                    embeddings_root=args.embeddings_root,
                    module="embeddings" if args.module == "penultimate" else "logits",
            )
        except:
            print("Could not load embeddings. Continuing without embeddings.")
            embeddings = None
        try:
            # Remove dicitonary level if it exists
            embeddings = embeddings[
                "embeddings" if args.module == "penultimate" else "logits"
            ]
        except:
            pass
    else:
        embeddings = None
    model_cfg, data_cfg = create_config_dicts(
            args, None
    )

    # Load transforms
    transforms = {
        source: {model_name: {} for model_name in model_cfg.names}
        for source in model_cfg.sources
    }
    if args.glob:
        args.alphas=[None]
        args.taus=[None]
        args.contrastive_batch_sizes=[None]
    eta, lmbda, alpha, tau, contrastive_batch_size = get_combination(
            etas=args.etas,
            lambdas=args.lmbdas,
            alphas=args.alphas,
            taus=args.taus,
            contrastive_batch_sizes=args.contrastive_batch_sizes,
    )
    for src, model_name in zip(model_cfg.sources, model_cfg.names):
        if args.glob:
            path_to_transform = os.path.join(args.transforms_root, model_name, model_cfg.module_type, "3", str(lmbda), args.optim.lower(), str(eta), "transform.npz")
            transforms[src][model_name] = GlobalTransform(
                    source=src,
                    model_name=model_name,
                    module=model_cfg.module_type,
                    path_to_transform=path_to_transform,
                    path_to_features=args.things_embeddings_path,
            )
        else:
            transforms[src][model_name] = GlocalTransform(
                    root=args.transforms_root,
                    source=src,
                    model=model_name,
                    module=model_cfg.module_type,
                    optim=args.optim.lower(),
                    eta=eta,
                    lmbda=lmbda,
                    alpha=alpha,
                    tau=tau,
                    contrastive_batch_size=contrastive_batch_size,
            )

    # Do few-shot
    all_results = []
    regressor_types = args.regressor_type
    n_shots = args.n_shot
    for model_id_in_cfg, (src, model_name, module) in enumerate(
            zip(model_cfg.sources, model_cfg.names, model_cfg.modules)
    ):
        for regressor_type in regressor_types:
            for shots in n_shots:
                if regressor_type == "ridge" and shots == 1:
                    continue
                args.n_shot = shots
                args.regressor_type = regressor_type
                model_cfg, data_cfg = create_config_dicts(
                        args, None
                )

                np.random.seed(int(1e5))
                torch.manual_seed(int(1e5))

                results = run(
                        n_shot=args.n_shot,
                        n_test=args.n_test,
                        n_reps=args.n_reps,
                        class_id_set=class_id_set,
                        class_id_set_test=class_id_set_test,
                        device=args.device,
                        model_cfg=model_cfg,
                        data_cfg=data_cfg,
                        transforms=transforms,
                        regressor_type=args.regressor_type,
                        superclass_mapping=superclass_mapping,
                        sample_per_superclass=args.sample_per_superclass,
                        model_id_in_cfg=model_id_in_cfg,
                        embeddings=embeddings,
                )
                all_results.append(results)
        results = pd.concat(all_results)
        results["lmbda"] = lmbda
        results["eta"] = eta
        results["optim"] = args.optim.lower()

        out_path = os.path.join(
                args.out_dir,
                args.dataset + ("" if args.task is None else f"_{args.task}"),
                model_cfg.sources[model_id_in_cfg],
                model_cfg.names[model_id_in_cfg],
                model_cfg.module_type,
                str(eta),
                str(lmbda),
                str(alpha),
                str(tau),
                str(contrastive_batch_size),
                str(args.sample_per_superclass),
                )
        if not os.path.exists(out_path):
            print("\nOutput directory does not exist...")
            print("Creating output directory to save results...\n")
            os.makedirs(out_path)

        results.to_pickle(os.path.join(out_path, "fewshot_results.pkl"))

    print("Elapsed time (init):", datetime.now() - start_t)
