import argparse
import os
import pickle
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from thingsvision import get_extractor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import data
import utils

NUM_WORKERS = 8

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/things")
    aa(
        "--imagenet_root",
        type=str,
        help="path/to/imagenet/data/folder",
        default="/home/space/datasets/imagenet/2012/",
    )
    aa("--dataset", type=str, help="Which dataset to use", default="things")
    aa("--model", type=str)
    aa(
        "--model_dict_path",
        type=str,
        default="/home/space/datasets/things/model_dict_all.json",
        help="Path to the model_dict.json",
    )
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
    aa("--learning_rate", type=float, metavar="eta", default=1e-3)
    aa(
        "--alpha",
        type=float,
        default=1e-1,
        help="Relative contribution of the classification loss term",
    )
    aa(
        "--lmbda",
        type=float,
        default=0,
        help="L2 regularization term",
    )
    aa(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient norm clipping value",
    )
    aa(
        "--triplet_batch_size",
        type=int,
        default=256,
        help="Use power of 2 for running optimization process on GPU",
    )
    aa(
        "--classification_batch_size",
        type=int,
        default=1024,
        help="Use power of 2 for running optimization process on GPU",
    )
    aa(
        "--epochs",
        type=int,
        help="Maximum number of epochs",
        default=100,
    )
    aa(
        "--patience",
        type=int,
        help="number of checks with no improvement after which training will be stopped",
        default=10,
    )
    aa("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    aa(
        "--num_processes",
        type=int,
        default=4,
        help="Number of devices to use for performing distributed training on CPU",
    )
    aa("--probing_root", type=str, help="path/to/probing")
    aa("--log_dir", type=str, help="directory to checkpoint transformations")
    aa("--rnd_seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()
    return args


def create_optimization_config(args) -> Dict[str, Any]:
    """Create frozen config dict for optimization hyperparameters."""
    optim_cfg = dict()
    optim_cfg["optim"] = args.optim
    optim_cfg["lr"] = args.learning_rate
    optim_cfg["alpha"] = args.alpha
    optim_cfg["lmbda"] = args.lmbda
    optim_cfg["classification_batch_size"] = args.classification_batch_size
    optim_cfg["triplet_batch_size"] = args.triplet_batch_size
    optim_cfg["max_epochs"] = args.epochs
    optim_cfg["patience"] = args.patience
    optim_cfg["ckptdir"] = os.path.join(args.log_dir, args.model, args.module)
    return optim_cfg


def create_model_config(args) -> Dict[str, Any]:
    """Create frozen config dict for optimization hyperparameters."""
    model_cfg = dict()
    model_config = utils.evaluation.load_model_config(args.model_dict_path)
    model_cfg["model"] = args.model
    model_cfg["module"] = model_config[args.model][args.module]["module_name"]
    model_cfg["source"] = args.source
    model_cfg["device"] = "cuda" if args.device == "gpu" else args.device
    return model_cfg


def load_features(probing_root: str, subfolder: str = "embeddings") -> Dict[str, Array]:
    """Load features for THINGS objects from disk."""
    with open(os.path.join(probing_root, subfolder, "features.pkl"), "rb") as f:
        features = pickle.load(f)
    return features


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
        monitor="val_loss",
        dirpath=optim_cfg["ckptdir"],
        filename="from-scratch-epoch{epoch:02d}-val_loss{val/loss:.2f}",
        auto_insert_metric_name=False,
        every_n_epochs=steps,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        mode="min",
        patience=optim_cfg["patience"],
        verbose=True,
        check_finite=True,
    )
    callbacks = [checkpoint_callback, early_stopping]
    return callbacks


def get_mean_cv_metric(cv_results: Dict[str, List[float]], metric: str) -> float:
    avg_val = np.mean([vals[0][metric] for vals in cv_results.values()])
    return avg_val


def save_results(
    args, imgnt_acc: float, imgnt_loss: float, things_acc: float, things_loss: float
) -> None:
    # Create dataframe with results
    probing_results = {
        "model": args.model_name,
        "imagenet_acc": imgnt_acc,
        "imagenet_loss": imgnt_loss,
        "things_acc": things_acc,
        "things_loss": things_loss,
        "module": args.module,
        "family": utils.analyses.get_family_name(args.model_name),
        "source": args.source,
        "optim": args.optim.lower(),
        "lr": args.learning_rate,
        "alpha": args.alpha,
        "lmbda": args.lmbda,
    }
    probing_results = pd.DataFrame({k: {0: v} for k, v in probing_results.items()})

    # Save results to disk
    out_path = os.path.join(args.probing_root, "results")
    outfile_path = os.path.join(out_path, "fromscratch_results.pkl")
    if not os.path.exists(out_path):
        print("\nCreating results directory...\n")
        os.makedirs(out_path)
    if os.path.isfile(outfile_path):
        print(
            "\nFile for probing results exists.\nConcatenating current results with existing results file...\n"
        )
        probing_results_overall = pd.read_pickle(outfile_path)
        probing_results = pd.concat(
            [probing_results_overall, probing_results],
            axis=0,
            ignore_index=True,
        )
    else:
        print("\nCreating file for probing results...\n")

    probing_results.to_pickle(outfile_path)


def load_extractor(model_cfg: Dict[str, str]) -> Any:
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
        pretrained=False,
        model_parameters=model_params,
    )
    return extractor


def run(
    imagenet_root: str,
    data_root: str,
    model_cfg: Dict[str, str],
    optim_cfg: Dict[str, Any],
    n_objects: int,
    device: str,
    rnd_seed: int,
    num_processes: int,
) -> Tuple[Dict[str, List[float]], Array]:
    """Run optimization process."""
    callbacks = get_callbacks(optim_cfg)
    extractor = load_extractor(model_cfg)
    """
    from thingsvision.utils.data import ImageDataset
    imagenet_train_set = ImageDataset(
            root=imagenet_root,
            out_path='./test_features',
            backend=extractor.get_backend(),
            transforms=extractor.get_transformations(resize_dim=256, crop_dim=224) # set input dimensionality to whatever is needed for your pretrained model
            )
    """
    # TODO: should we use thingsvision here or not? -- we need labels
    imagenet_train_set = ImageFolder(
        os.path.join(imagenet_root, "train_set"),
        extractor.get_transformations(resize_dim=256, crop_dim=224),
    )
    imagenet_val_set = ImageFolder(
        os.path.join(imagenet_root, "val_set"),
        extractor.get_transformations(resize_dim=256, crop_dim=224),
    )
    triplets = utils.probing.load_triplets(data_root)
    objects = np.arange(n_objects)
    # We don't need to perform k-Fold cross-validation (we can simply set k=4 or 5)
    kf = KFold(n_splits=4, random_state=rnd_seed, shuffle=True)
    cv_results = {}
    ooo_choices = []
    for k, (train_idx, _) in tqdm(enumerate(kf.split(objects), start=1), desc="Fold"):
        train_objects = objects[train_idx]
        # partition triplets into disjoint object sets
        triplet_partitioning = utils.probing.partition_triplets(
            triplets=triplets,
            train_objects=train_objects,
        )
        """
        train_triplets = utils.probing.TripletData(
            triplets=triplet_partitioning["train"],
            n_objects=n_objects,
        )
        val_triplets = utils.probing.TripletData(
            triplets=triplet_partitioning["val"],
            n_objects=n_objects,
        )
        """
        # TODO: are those the right transformations? & are we using -aligned- triplets?
        train_triplets = data.THINGSTriplet(root=data_root, transform=extractor.get_transformations())
        train_triplets.triplets = np.array(triplet_partitioning["train"])
        val_triplets = data.THINGSTriplet(root=data_root, transform=extractor.get_transformations())
        val_triplets.triplets = np.array(triplet_partitioning["val"])

        train_batches_things = get_batches(
            dataset=train_triplets,
            batch_size=optim_cfg["triplet_batch_size"],
            train=True,
            num_workers=NUM_WORKERS,
        )
        train_batches_imagenet = get_batches(
            dataset=imagenet_train_set,
            batch_size=optim_cfg["classification_batch_size"],
            train=True,
            num_workers=NUM_WORKERS,
        )
        val_batches_things = get_batches(
            dataset=val_triplets,
            batch_size=optim_cfg["triplet_batch_size"],
            train=False,
            num_workers=NUM_WORKERS,
        )
        val_batches_imagenet = get_batches(
            dataset=imagenet_val_set,
            batch_size=optim_cfg["classification_batch_size"],
            train=True,  # TODO ?
            num_workers=NUM_WORKERS,
        )
        train_batches = utils.probing.zip_batches(
            train_batches_things, train_batches_imagenet
        )
        val_batches = utils.probing.zip_batches(
            val_batches_things, val_batches_imagenet
        )
        trainable = utils.probing.FromScratch(
            optim_cfg=optim_cfg,
            model_cfg=model_cfg,
            extractor=extractor,
        )
        trainer = Trainer(
            accelerator=device,
            callbacks=callbacks,
            strategy="ddp",
            max_epochs=optim_cfg["max_epochs"],
            devices=num_processes if device == "cpu" else "auto",
            enable_progress_bar=True,
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm="norm",
            precision=16 if device == "gpu" else 32,
        )

        trainer.fit(trainable, train_batches, val_batches)
        val_performance = trainer.test(
            trainable,
            dataloaders=val_batches,
        )
        # Reset val batches to get predictions
        val_batches = utils.probing.zip_batches(
            val_batches_things, val_batches_imagenet
        )
        predictions = trainer.predict(trainable, dataloaders=val_batches)
        predictions = torch.cat(predictions, dim=0).tolist()
        ooo_choices.append(predictions)
        cv_results[f"fold_{k:02d}"] = val_performance
        break
    model = trainable.model
    ooo_choices = np.concatenate(ooo_choices)
    return ooo_choices, cv_results, model


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    # seed everything for reproducibility of results
    seed_everything(args.rnd_seed, workers=True)
    # run optimization
    optim_cfg = create_optimization_config(args)
    model_cfg = create_model_config(args)
    ooo_choices, cv_results, model = run(
        imagenet_root=args.imagenet_root,
        data_root=args.data_root,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        n_objects=args.n_objects,
        device=args.device,
        rnd_seed=args.rnd_seed,
        num_processes=args.num_processes,
    )
    avg_cv_imgnt_acc = get_mean_cv_metric(cv_results, "test_imgnt_acc")
    avg_cv_imgnt_loss = get_mean_cv_metric(cv_results, "test_imgnt_loss")
    avg_cv_things_acc = get_mean_cv_metric(cv_results, "test_things_acc")
    avg_cv_things_loss = get_mean_cv_metric(cv_results, "test_things_loss")
    # save results
    save_results(
        args,
        imgnt_acc=avg_cv_imgnt_acc,
        imgnt_loss=avg_cv_imgnt_loss,
        things_acc=avg_cv_things_acc,
        things_loss=avg_cv_things_loss,
    )
    # save model
    out_path = os.path.join(
        args.probing_root,
        "results",
        args.source,
        args.model,
        args.module,
        str(args.lmbda),
        args.optim.lower(),
        str(args.learning_rate),
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    model_save_path = os.path.join(out_path, "model.pt")
    torch.save(model.state_dict(), model_save_path)
