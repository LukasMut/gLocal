import argparse
import itertools
import os
import pickle
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/things")
    aa("--imagenet_features_root", type=str, help="path/to/imagenet/features")
    aa("--dataset", type=str, help="Which dataset to use", default="things")
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
        default="torchvision",
        choices=[
            "google",
            "loss",
            "custom",
            "ssl",
            "imagenet",
            "torchvision",
            "vit_same",
            "vit_best",
        ],
    )
    aa(
        "--n_objects",
        type=int,
        help="Number of object categories in the data",
        default=1854,
    )
    aa("--optim", type=str, default="Adam", choices=["Adam", "AdamW", "SGD"])
    aa(
        "--learning_rates",
        type=float,
        default=1e-3,
        nargs="+",
        metavar="eta",
        choices=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    )
    aa(
        "--regularization",
        type=str,
        default="l2",
        choices=["l2", "eye"],
        help="What kind of regularization to be applied",
    )
    aa(
        "--lmbdas",
        type=float,
        default=1e-3,
        nargs="+",
        help="Relative contribution of the l2 or identity regularization penality",
        choices=[10.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
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
        choices=[1.0, 5e-1, 25e-2, 1e-1, 5e-2, 25e-3, 1e-2],
    )
    aa(
        "--sigma",
        type=float,
        default=1e-3,
        help="Scalar to scale a neural net's pre-transformed representation space prior to the optimization process",
        choices=[1.0, 1e-1, 1e-2, 1e-3, 1e-4],
    )
    aa(
        "--triplet_batch_size",
        type=int,
        default=256,
        metavar="B_T",
        help="Use 64 <= B <= 1024 and power of two for running optimization process on GPU",
        choices=[64, 128, 256, 512, 1024],
    )
    aa(
        "--contrastive_batch_sizes",
        type=int,
        default=1024,
        nargs="+",
        metavar="B_C",
        help="Use 64 <= B <= 4096 and power of two for running optimization process on GPU",
        choices=[64, 128, 256, 512, 1024, 2048, 4096],
    )
    aa(
        "--epochs",
        type=int,
        help="Maximum number of epochs to perform finetuning",
        default=100,
        choices=[50, 100, 150, 200, 250, 300],
    )
    aa(
        "--burnin",
        type=int,
        help="Minimum number of epochs to perform finetuning",
        default=20,
    )
    aa(
        "--patience",
        type=int,
        help="number of checks with no improvement after which training will be stopped",
        default=15,
    )
    aa("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    aa(
        "--features_format",
        type=str,
        default="hdf5",
        help="In which data format ImageNet features have been saved to disk",
        choices=["hdf5", "pt"],
    )
    aa(
        "--num_processes",
        type=int,
        default=4,
        choices=[2, 4, 6, 8, 10, 12],
        help="Number of devices to use for performing distributed training on CPU",
    )
    aa(
        "--use_bias",
        action="store_true",
        help="whether to use a bias in the linear probe",
    )
    aa(
        "--adversarial",
        action="store_true",
        help="whether to use adversarial triplets for training",
    )
    aa("--probing_root", type=str, help="path/to/probing")
    aa("--log_dir", type=str, help="directory to checkpoint transformations")
    aa("--rnd_seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()
    return args


def get_combination(
    etas: List[float],
    lambdas: List[float],
    alphas: List[float],
    taus: List[float],
    contrastive_batch_sizes: List[int],
) -> List[Tuple[float, float, float, float, int]]:
    combs = []
    combs.extend(
        list(itertools.product(etas, lambdas, alphas, taus, contrastive_batch_sizes))
    )
    return combs[int(os.environ["SLURM_ARRAY_TASK_ID"])]


def create_optimization_config(
    args,
    eta: float,
    lmbda: float,
    alpha: float,
    tau: float,
    contrastive_batch_size: int,
    out_path: str,
) -> Dict[str, Any]:
    """Create frozen config dict for optimization hyperparameters."""
    optim_cfg = dict()
    optim_cfg["optim"] = args.optim
    optim_cfg["reg"] = args.regularization
    optim_cfg["lr"] = eta
    optim_cfg["lmbda"] = lmbda
    optim_cfg["alpha"] = alpha
    optim_cfg["tau"] = tau
    optim_cfg["contrastive_batch_size"] = contrastive_batch_size
    optim_cfg["triplet_batch_size"] = args.triplet_batch_size
    optim_cfg["max_epochs"] = args.epochs
    optim_cfg["min_epochs"] = args.burnin
    optim_cfg["patience"] = args.patience
    optim_cfg["use_bias"] = args.use_bias
    optim_cfg["ckptdir"] = os.path.join(args.log_dir, args.model, args.module)
    optim_cfg["sigma"] = args.sigma
    optim_cfg["adversarial"] = args.adversarial
    optim_cfg["out_path"] = out_path
    return optim_cfg


def load_features(probing_root: str, subfolder: str = "embeddings") -> Dict[str, Array]:
    """Load features for THINGS objects from disk."""
    with open(os.path.join(probing_root, subfolder, "features.pkl"), "rb") as f:
        features = pickle.load(f)
    return features


def get_temperature(
    model_config, model: List[str], module: str, objective: str = "cosine"
) -> List[str]:
    """Get optimal temperature values for all models."""
    try:
        temp = model_config[model][module]["temperature"][objective]
    except KeyError:
        temp = 1.0
        warnings.warn(
            f"\nMissing temperature value for {model} and {module} layer.\nSetting temperature value to 1.\n"
        )
    return temp


def get_batches(
    dataset: Tensor, batch_size: int, train: bool, num_workers: int = 0
) -> Iterator:
    batches = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if train else False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True if train else False,
    )
    return batches


def get_callbacks(optim_cfg: FrozenDict, steps: int = 20) -> List[Callable]:
    if not os.path.exists(optim_cfg["ckptdir"]):
        os.makedirs(optim_cfg["ckptdir"])
        print("\nCreating directory for checkpointing...\n")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_overall_loss",
        dirpath=optim_cfg["ckptdir"],
        filename="ooo-finetuning-epoch{epoch:02d}-val_overall_loss{val/overall_loss:.2f}",
        auto_insert_metric_name=False,
        every_n_epochs=steps,
    )
    early_stopping = EarlyStopping(
        monitor="val_overall_loss",
        min_delta=1e-4,
        mode="min",
        patience=optim_cfg["patience"],
        verbose=True,
        check_finite=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, early_stopping, lr_monitor]
    return callbacks


def get_mean_cv_performances(cv_results: Dict[str, List[float]]) -> Dict[str, float]:
    return {metric: np.mean(folds) for metric, folds in cv_results.items()}


def make_results_df(
    columns: List[str],
    probing_performances: Dict[str, float],
    ooo_choices: Array,
    model_name: str,
    module_name: str,
    source: str,
    optim_cfg: Dict[str, Any],
) -> pd.DataFrame:
    probing_results_current_run = pd.DataFrame(index=range(1), columns=columns)
    probing_results_current_run["model"] = model_name
    probing_results_current_run["probing"] = probing_performances["test_acc"]
    probing_results_current_run["cross-entropy-overall"] = probing_performances[
        "test_overall_loss"
    ]
    probing_results_current_run["triplet-loss"] = probing_performances[
        "test_triplet_loss"
    ]
    probing_results_current_run["locality-loss"] = probing_performances[
        "test_contrastive_loss"
    ]
    # probing_results_current_run["choices"] = [ooo_choices]
    probing_results_current_run["module"] = module_name
    try:
        probing_results_current_run["family"] = utils.analyses.get_family_name(
            model_name
        )
    except UnboundLocalError:
        probing_results_current_run[
            "family"
        ] = model_name  # NOTE: this is a hack to get the family name for the DINO models
    probing_results_current_run["source"] = source
    probing_results_current_run["reg"] = optim_cfg["reg"]
    probing_results_current_run["optim"] = optim_cfg["optim"].lower()
    probing_results_current_run["lr"] = optim_cfg["lr"]
    probing_results_current_run["alpha"] = optim_cfg["alpha"]
    probing_results_current_run["lmbda"] = optim_cfg["lmbda"]
    probing_results_current_run["tau"] = optim_cfg["tau"]
    probing_results_current_run["sigma"] = optim_cfg["sigma"]
    probing_results_current_run["bias"] = optim_cfg["use_bias"]
    probing_results_current_run["contrastive_batch_size"] = optim_cfg[
        "contrastive_batch_size"
    ]
    probing_results_current_run["triplet_batch_size"] = optim_cfg["triplet_batch_size"]
    probing_results_current_run["contrastive"] = True
    return probing_results_current_run


def save_results(
    args,
    optim_cfg: Dict[str, Any],
    probing_performances: Dict[str, float],
    ooo_choices: Array,
) -> None:
    out_path = os.path.join(args.probing_root, "results")
    if not os.path.exists(out_path):
        print("\nCreating results directory...\n")
        os.makedirs(out_path)

    if os.path.isfile(os.path.join(out_path, "probing_results.pkl")):
        print(
            "\nFile for probing results exists.\nConcatenating current results with existing results file...\n"
        )
        probing_results_overall = pd.read_pickle(
            os.path.join(out_path, "probing_results.pkl")
        )
        probing_results_current_run = make_results_df(
            columns=probing_results_overall.columns.values,
            probing_performances=probing_performances,
            ooo_choices=ooo_choices,
            model_name=args.model,
            module_name=args.module,
            source=args.source,
            optim_cfg=optim_cfg,
        )
        probing_results = pd.concat(
            [probing_results_overall, probing_results_current_run],
            axis=0,
            ignore_index=True,
        )
        probing_results.to_pickle(os.path.join(out_path, "probing_results.pkl"))
    else:
        print("\nCreating file for probing results...\n")
        columns = [
            "model",
            "probing",
            "cross-entropy-overall",
            "triplet-loss",
            "locality-loss",
            # "choices",
            "module",
            "family",
            "source",
            "reg",
            "optim",
            "lr",
            "alpha",
            "lambda",
            "tau",
            "sigma",
            "bias",
            "contrastive_batch_size",
            "triplet_batch_size",
            "contrastive",
        ]
        probing_results = make_results_df(
            columns=columns,
            probing_performances=probing_performances,
            ooo_choices=ooo_choices,
            model_name=args.model,
            module_name=args.module,
            source=args.source,
            optim_cfg=optim_cfg,
        )
        probing_results.to_pickle(os.path.join(out_path, "probing_results.pkl"))


def get_imagenet_features(root: str, format: str, device: str) -> Tuple[Any, Any]:
    if format == "hdf5":
        imagenet_train_features = utils.probing.FeaturesHDF5(
            root=root,
            split="train",
        )
        imagenet_val_features = utils.probing.FeaturesHDF5(
            root=root,
            split="val",
        )
    elif format == "pt":
        imagenet_train_features = utils.probing.FeaturesPT(
            root=root,
            split="train",
            device=device,
        )
        imagenet_val_features = utils.probing.FeaturesPT(
            root=root,
            split="val",
            device=device,
        )
    else:
        raise ValueError(
            "\nTensor datasets can only be created for features that were saved in either 'pt' or 'hdf5' format.\n"
        )
    return imagenet_train_features, imagenet_val_features


def run(
    features: Array,
    imagenet_features_root: str,
    data_root: str,
    optim_cfg: Dict[str, Any],
    n_objects: int,
    device: str,
    rnd_seed: int,
    num_processes: int,
    features_format: str,
) -> Tuple[Dict[str, List[float]], Array]:
    """Run optimization process."""
    callbacks = get_callbacks(optim_cfg)
    imagenet_train_features, imagenet_val_features = get_imagenet_features(
        root=imagenet_features_root,
        format=features_format,
        device="cuda" if device == "gpu" else device,
    )
    triplets = utils.probing.load_triplets(
        data_root=data_root, adversarial=optim_cfg["adversarial"]
    )
    things_mean = features.mean()
    things_std = features.std()
    features = (
        features - things_mean
    ) / things_std  # subtract global mean and normalize by standard deviation of feature matrix
    optim_cfg["things_mean"] = things_mean
    optim_cfg["things_std"] = things_std
    objects = np.arange(n_objects)
    # For glocal optimization, we don't need to perform k-Fold cross-validation (we can simply set k=4 or 5)
    kf = KFold(n_splits=4, random_state=rnd_seed, shuffle=True)
    cv_results = defaultdict(list)
    ooo_choices = []
    for train_idx, _ in tqdm(kf.split(objects), desc="Fold"):
        train_objects = objects[train_idx]
        # partition triplets into disjoint object sets
        triplet_partitioning = utils.probing.partition_triplets(
            triplets=triplets,
            train_objects=train_objects,
        )
        train_triplets = utils.probing.TripletData(
            triplets=triplet_partitioning["train"],
            n_objects=n_objects,
        )
        val_triplets = utils.probing.TripletData(
            triplets=triplet_partitioning["val"],
            n_objects=n_objects,
        )
        train_batches_things = get_batches(
            dataset=train_triplets,
            batch_size=optim_cfg["triplet_batch_size"],
            train=True,
            num_workers=0,
        )
        train_batches_imagenet = get_batches(
            dataset=imagenet_train_features,
            batch_size=optim_cfg["contrastive_batch_size"],
            train=True,
            num_workers=num_processes,
        )
        val_batches_things = get_batches(
            dataset=val_triplets,
            batch_size=optim_cfg["triplet_batch_size"],
            train=False,
            num_workers=0,
        )
        val_batches_imagenet = get_batches(
            dataset=imagenet_val_features,
            batch_size=optim_cfg["contrastive_batch_size"],
            train=True,
            num_workers=num_processes,
        )
        train_batches = utils.probing.ZippedBatchLoader(
            batches_i=train_batches_things,
            batches_j=train_batches_imagenet,
            num_workers=num_processes,
        )
        val_batches = utils.probing.ZippedBatchLoader(
            batches_i=val_batches_things,
            batches_j=val_batches_imagenet,
            num_workers=num_processes,
        )
        glocal_probe = utils.probing.GlocalFeatureProbe(
            features=features,
            optim_cfg=optim_cfg,
        )
        trainer = Trainer(
            accelerator=device,
            callbacks=callbacks,
            strategy="ddp_spawn" if device == "cpu" else None,
            # strategy="ddp",
            max_epochs=optim_cfg["max_epochs"],
            min_epochs=optim_cfg["min_epochs"],
            devices=num_processes if device == "cpu" else "auto",
            enable_progress_bar=True,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )
        trainer.fit(glocal_probe, train_batches, val_batches)
        test_performances = trainer.test(
            glocal_probe,
            dataloaders=val_batches,
        )
        predictions = trainer.predict(glocal_probe, dataloaders=val_batches_things)
        predictions = torch.cat(predictions, dim=0).tolist()
        ooo_choices.append(predictions)
        for metric, performance in test_performances[0].items():
            cv_results[metric].append(performance)
        break
    transformation = {}
    transformation["weights"] = glocal_probe.transform_w.data.detach().cpu().numpy()
    if optim_cfg["use_bias"]:
        transformation["bias"] = glocal_probe.transform_b.data.detach().cpu().numpy()
    ooo_choices = np.concatenate(ooo_choices)
    return ooo_choices, cv_results, transformation, things_mean, things_std


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    # seed everything for reproducibility of results
    seed_everything(args.rnd_seed, workers=True)
    features = load_features(args.probing_root)
    model_features = features[args.source][args.model][args.module]

    eta, lmbda, alpha, tau, contrastive_batch_size = get_combination(
        etas=args.learning_rates,
        lambdas=args.lmbdas,
        alphas=args.alphas,
        taus=args.taus,
        contrastive_batch_sizes=args.contrastive_batch_sizes,
    )

    out_path = os.path.join(
        args.probing_root,
        "results",
        args.source,
        args.model,
        args.module,
        args.optim.lower(),
        str(eta),
        str(lmbda),
        str(alpha),
        str(tau),
        str(contrastive_batch_size),
    )
    if args.adversarial:
        out_path = os.path.join(out_path, "adversarial")

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    out_file_path = os.path.join(out_path, "transform.npz")

    if os.path.isfile(out_file_path):
        print("Results already exist. Skipping...")
        print(f"Results file: {out_file_path}")
    else:
        optim_cfg = create_optimization_config(
            args=args,
            eta=eta,
            lmbda=lmbda,
            alpha=alpha,
            tau=tau,
            contrastive_batch_size=contrastive_batch_size,
            out_path=out_path,
        )

        ooo_choices, cv_results, transform, things_mean, things_std = run(
            features=model_features,
            imagenet_features_root=args.imagenet_features_root,
            data_root=args.data_root,
            optim_cfg=optim_cfg,
            n_objects=args.n_objects,
            device=args.device,
            rnd_seed=args.rnd_seed,
            num_processes=args.num_processes,
            features_format=args.features_format,
        )
        probing_performances = get_mean_cv_performances(cv_results)
        if optim_cfg["use_bias"]:
            with open(out_file_path, "wb") as f:
                np.savez_compressed(
                    file=f,
                    weights=transform["weights"],
                    bias=transform["bias"],
                    mean=things_mean,
                    std=things_std,
                )
        else:
            with open(out_file_path, "wb") as f:
                np.savez_compressed(
                    file=f,
                    weights=transform["weights"],
                    mean=things_mean,
                    std=things_std,
                )
        save_results(
            args=args,
            optim_cfg=optim_cfg,
            probing_performances=probing_performances,
            ooo_choices=ooo_choices,
        )
