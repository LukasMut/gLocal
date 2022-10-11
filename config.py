#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

from ml_collections import config_dict

from utils import get_class_weights


def get_configs(args, **kwargs):
    """Create config dicts for dataset, model and optimizer."""
    data_config = config_dict.ConfigDict()
    data_config.root = args.data_path
    data_config.name = args.dataset.lower()
    # balanced (homogeneous) or imbalanced (heterogeneous) dataset
    data_config.distribution = args.distribution
    # dataset imbalance is a function of p
    data_config.class_probs = 0.8
    # minimum number of instances per class
    data_config.min_samples = args.min_samples
    # whether to balance mini-batches
    data_config.sampling = args.sampling
    # input dimensionality
    data_config.input_dim = kwargs.pop("input_dim")
    # average number of instances per class
    M = kwargs.pop("n_samples")
    data_config.n_samples = M
    data_config.batch_size = kwargs.pop("batch_size")

    model_config = config_dict.ConfigDict()
    model_config.type = re.compile(r"[a-zA-Z]+").search(args.network).group()

    try:
        model_config.depth = re.compile(r"\d+").search(args.network).group()
    except AttributeError:
        model_config.depth = ""

    model_config.weight_decay = 1e-3
    model_config.sparsity_strength = 1e-2
    model_config.n_classes = args.n_classes
    model_config.task = args.task
    model_config.device = args.device
    
    # TODO: enable half precision when running things on TPU
    model_config.half_precision = False

    if args.mle_loss == "weighted":
        w = get_class_weights(M, args.n_classes, args.min_samples)
        model_config.weights = w
    else:
        model_config.weights = None

    optimizer_config = config_dict.ConfigDict()
    optimizer_config.name = args.optim
    optimizer_config.burnin = args.burnin
    optimizer_config.patience = args.patience
    optimizer_config.lr = kwargs.pop("eta")
    optimizer_config.epochs = kwargs.pop("epochs")

    if optimizer_config.name.lower() == "sgd":
        # add momentum param if optim is sgd
        optimizer_config.momentum = 0.9

    # make config dicts immutable (same type as model param dicts)
    freeze = lambda cfg: config_dict.FrozenConfigDict(cfg)
    # freeze = lambda cfg: flax.core.frozen_dict.FrozenDict(cfg)
    data_config = freeze(data_config)
    model_config = freeze(model_config)
    optimizer_config = freeze(optimizer_config)
    return data_config, model_config, optimizer_config
