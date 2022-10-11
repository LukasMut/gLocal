#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import warnings
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from ml_collections import config_dict

from data import DataLoader
from models import AugmentationNet
from training import Trainer

Array = np.ndarray
FrozenDict = config_dict.FrozenConfigDict


def get_combination(
    samples: List[int],
    epochs: List[int],
    batch_sizes: List[int],
    learning_rates: List[float],
    seeds: List[int],
):
    combs = []
    combs.extend(
        list(
            itertools.product(zip(samples, epochs, batch_sizes, learning_rates), seeds)
        )
    )
    # NOTE: for SLURM use "SLURM_ARRAY_TASK_ID"
    return combs[0]  # combs[int(os.environ["SGE_TASK_ID"])]


def make_path(
    root: str,
    model_config: FrozenDict,
    data_config: FrozenDict,
    rnd_seed: int,
) -> str:
    path = os.path.join(
        root,
        data_config.name,
        model_config.task,
        model_config.type + model_config.depth,
        f"{data_config.n_samples}_samples",
        data_config.distribution,
        f"seed{rnd_seed:02d}",
    )
    return path


def create_dirs(
    results_root: str,
    data_config: FrozenDict,
    model_config: FrozenDict,
    rnd_seed: int,
):
    """Create directories for saving and loading model checkpoints."""
    dir_config = config_dict.ConfigDict()
    log_dir = make_path(results_root, model_config, data_config, rnd_seed)
    dir_config.log_dir = log_dir

    if not os.path.exists(log_dir):
        print("\n...Creating directory to store model checkpoints.\n")
        os.makedirs(log_dir, exist_ok=True)

    return dir_config


def run(
    model,
    model_config: FrozenDict,
    data_config: FrozenDict,
    optimizer_config: FrozenDict,
    dir_config: FrozenDict,
    train_set: Tuple[Array, Array],
    val_set: Tuple[Array, Array],
    steps: int,
    rnd_seed: int,
) -> tuple:
    trainer = Trainer(
        model=model,
        model_config=model_config,
        data_config=data_config,
        optimizer_config=optimizer_config,
        dir_config=dir_config,
        steps=steps,
        rnd_seed=rnd_seed,
    )
    train_batches = DataLoader(
        data=train_set,
        data_config=data_config,
        model_config=model_config,
        seed=rnd_seed,
        train=True,
    )
    val_batches = DataLoader(
        data=val_set,
        data_config=data_config,
        model_config=model_config,
        seed=rnd_seed,
        train=False,
    )
    metrics, epoch = trainer.train(train_batches, val_batches)
    return trainer, metrics, epoch


def inference(
    out_path: str,
    trainer: object,
    X_test: Array,
    y_test: Array,
    train_labels: Array,
    model_config: FrozenDict,
    data_config: FrozenDict,
) -> None:
    try:
        loss, cls_hits = trainer.inference(X_test, y_test)
    except (RuntimeError, MemoryError):
        warnings.warn(
            "\nTest set does not fit into the GPU's memory.\nSplitting test set into small batches to counteract memory problems.\n"
        )
        loss, cls_hits = trainer.batch_inference(X_test, y_test)
    acc = {cls: np.mean(hits) for cls, hits in cls_hits.items()}
    test_performance = config_dict.FrozenConfigDict({"loss": loss, "accuracy": acc})
    train_labels = np.nonzero(train_labels, size=train_labels.shape[0])[-1]
    cls_distribution = dict(Counter(train_labels.tolist()))

    print(test_performance)
    print()

    save_results(
        out_path=out_path,
        performance=test_performance,
        cls_distribution=cls_distribution,
        model_config=model_config,
        data_config=data_config,
    )


def make_results_df(
    columns: List[str],
    performance: FrozenDict,
    cls_distribution: Dict[int, int],
    model_config: FrozenDict,
    data_config: FrozenDict,
) -> pd.DataFrame:
    results_current_run = pd.DataFrame(index=range(1), columns=columns)
    results_current_run["model"] = model_config.type + model_config.depth
    results_current_run["dataset"] = data_config.name
    results_current_run["class-distribution"] = [cls_distribution]
    results_current_run["class-performance"] = [list(performance["accuracy"].items())]
    results_current_run["avg-performance"] = np.mean(
        list(performance["accuracy"].values())
    )
    results_current_run["cross-entropy"] = performance["loss"]
    results_current_run["training"] = model_config.task
    results_current_run["n_samples"] = data_config.n_samples
    results_current_run["probability"] = data_config.class_probs
    return results_current_run


def save_results(
    out_path: str,
    performance: FrozenDict,
    cls_distribution: Dict[int, int],
    model_config: FrozenDict,
    data_config: FrozenDict,
) -> None:
    if not os.path.exists(out_path):
        print("\nCreating results directory...\n")
        os.makedirs(out_path, exist_ok=True)

    if os.path.isfile(os.path.join(out_path, "results.pkl")):
        print(
            "\nFile for results exists.\nConcatenating current results with existing results file...\n"
        )
        results_overall = pd.read_pickle(os.path.join(out_path, "results.pkl"))
        results_current_run = make_results_df(
            columns=results_overall.columns.values,
            performance=performance,
            cls_distribution=cls_distribution,
            model_config=model_config,
            data_config=data_config,
        )
        results = pd.concat(
            [results_overall, results_current_run], axis=0, ignore_index=True
        )
        results.to_pickle(os.path.join(out_path, "results.pkl"))
    else:
        print("\nCreating file for results...\n")
        columns = [
            "model",
            "dataset",
            "class-distribution",
            "class-performance",
            "avg-performance",
            "cross-entropy",
            "training",
            "n_samples",
            "probability",
        ]
        results_current_run = make_results_df(
            columns=columns,
            performance=performance,
            cls_distribution=cls_distribution,
            model_config=model_config,
            data_config=data_config,
        )
        results_current_run.to_pickle(os.path.join(out_path, "results.pkl"))


def get_model(model_config: FrozenDict, data_config: FrozenDict):
    """Create model instance."""
    model_name = model_config.type + model_config.depth
    model = AugmentationNet(
        backbone=model_name,
        num_classes=model_config.n_classes,
        grayscale=True if data_config.name.lower().endswith("mnist") else False,
    )
    return model
