#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import re
import visualization

import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from typing import Dict, Tuple

os.environ['PYTHONIOENCODING']='UTF-8'
os.environ['OMP_NUM_THREADS']='1'


def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--in_path', type=str,
        help='path/to/results')
    aa('--out_path', type=str,
        help='path/to/plots')
    aa('--dataset', type=str, default='mnist',
        choices=['mnist', 'imagenet', 'real-world'])
    aa('--distribution', type=str,
        choices=['homogeneous', 'heterogeneous'],
        help='whether class distribution is uniform or non-uniform')
    aa('--metric', type=str,
        choices=['accuracy', 'cross-entropy'])
    aa('--verbose', action='store_true',
        help='whether or not to show plot')
    args = parser.parse_args()
    return args


def get_results(
                PATH: str,
                dist: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    results = defaultdict(lambda: defaultdict(dict))
    for root, _, files in os.walk(PATH, followlinks=True):
        for f in files:
            if re.compile(r'(?=^performance)(?=.*pkl$)').search(f):
                path_split = root.split('/')
                distribution = path_split[4]
                if distribution == dist:
                    n_samples = path_split[3]
                    n_samples = int(re.compile(r'\d+').search(n_samples).group())
                    seed = path_split[5]
                    training = path_split[7]
                    if training == 'pretraining':
                        if re.compile(r'frozen').search(root):
                            training += '_frozen'
                    performance = pickle.loads(open(os.path.join(root, f), 'rb').read())
                    try:
                        results[training][n_samples][seed]['accuracy'] = performance['accuracy']
                        results[training][n_samples][seed]['cross-entropy'] = performance['loss']
                    except KeyError:
                        results[training][n_samples][seed] = {}
                        results[training][n_samples][seed]['accuracy'] = performance['accuracy']
                        results[training][n_samples][seed]['cross-entropy'] = performance['loss']
    results = sort_results(results)
    return results


def sort_results(
                results: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    def sort_dict(splits: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
        return dict(sorted(splits.items(), key=lambda kv:kv[0], reverse=False))
    return {training: sort_dict(splits) for training, splits in results.items()}


def dict2df(
            results: Dict[str, Dict[str, Dict[str, float]]],
            metric: str,
) -> pd.DataFrame:
    n_rows = range(len([val[metric]
                        for split in results.values()
                        for initializations in split.values()
                        for val in initializations.values()]))
    results_df = pd.DataFrame(
        columns=['samples', 'performance', 'training'],
        index=n_rows, dtype=float)
    i = 0
    for training, split in results.items():
        for n_samples, initializations in split.items():
            for val in initializations.values():
                results_df.loc[i, 'samples'] = n_samples
                if metric == 'accuracy':
                    performance = val[metric] * 100
                else:
                    performance = val[metric]
                results_df.loc[i, 'performance'] = performance
                if re.search(r'^ml', training):
                    results_df.loc[i, 'training'] = 'MLE'
                elif re.search(r'frozen$', training):
                    results_df.loc[i, 'training'] = 'OOO (frozen)'
                else:
                    results_df.loc[i, 'training'] = 'OOO'
                i += 1
    return results_df


def prepare_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    samples = df.samples.unique().astype(int)
    spaces = np.linspace(0, samples.max(), len(samples))
    df['samples'].replace({num: el for num, el in zip(samples, spaces)}, inplace=True)
    return df, samples


def rearrange_results(results: pd.DataFrame) -> pd.DataFrame:
    subsets = []
    for training in results.training.unique():
        subset = results[results['training'] == training]
        subset = subset.rename({'performance': training}, axis='columns')
        subset = subset.drop(['training'], axis=1)
        subset = subset.reset_index(drop=True)
        subsets.append(subset)
    results = subsets.pop(0)
    for subset in subsets:
        results = results.join(subset[subset.columns[1]])
    results['samples'] = results['samples'].apply(int)
    return results


if __name__ == '__main__':
    # parse arguments
    args = parseargs()

    # create directory to save plots
    out_path = os.path.join(args.out_path, args.dataset, args.distribution, args.metric)
    if not os.path.exists(out_path):
        print('\n...Creating directories to save figure.\n')
        os.makedirs(out_path)

    # get results
    results = get_results(args.in_path, args.distribution)
    df = dict2df(results, args.metric)
    df = df[df['samples'] != 500]

    if metric == 'accuracy':
        rearranged_df = rearrange_results(df)
        visualization.plot_scatters(
                                    results=rearranged_df,
                                    dataset=args.dataset,
                                    out_path=out_path,
                                    verbose=args.verbose,
        )

    df, samples = prepare_df(df)

    # plot results
    visualization.plot_lines(
                            results=df,
                            samples=samples,
                            metric=args.metric,
                            dataset=args.dataset,
                            out_path=out_path,
                            verbose=args.verbose,
                            )