#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from typing import Dict


def plot_scatters(
                    results: pd.DataFrame,
                    dataset: str,
                    out_path: str,
                    verbose: bool=True,
) -> None:    
    combs = list(itertools.combinations(results.columns[1:], 2))
    # sequential blue color palette
    # color_palette = sns.dark_palette("#69d", reverse=True, as_cmap=True)
    # sequential red color palette
    color_palette = sns.color_palette("dark:salmon_r", as_cmap=True)
    regex = r'frozen'
    it_label = 'OOO $\it{(frozen)}$'
    for comb in combs: 
        comb_0_performance = results[comb[0]]
        comb_1_performance = results[comb[1]]
        plt.figure(figsize=(8, 4), dpi=80)
        ax = sns.scatterplot(
                            data=results,
                            x=comb[1],
                            y=comb[0],
                            hue="samples",
                            size="samples",
                            legend="full",
                            alpha=0.7,
                            sizes=np.linspace(30, 70, len(results.samples.unique())).tolist(),
                            palette=color_palette,
        )
        ax.plot([-5, 110], [-5, 110], ls="--", c=".3")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        
        ax.set_xlim(-5, 110)
        ax.set_ylim(-5, 110)
        
        ax.set_xticks(np.arange(0, 110, 20))
        ax.set_yticks(np.arange(0, 110, 20))
        
        if re.compile(regex).search(comb[1]):
            xlab = it_label
        else:
            xlab = comb[1]
        
        if re.compile(regex).search(comb[0]):
            ylab = it_label
        else:
            ylab = comb[0]
            
        ax.set_xlabel(xlab, fontsize=13, labelpad=10)
        ax.set_ylabel(ylab, fontsize=13, labelpad=10)
        ax.set_title(dataset.upper())
        
        ax.legend().set_title('')
        ax.legend(shadow=True, fancybox=True, loc='upper left', fontsize=8)

        plt.tight_layout()
        
        plt.savefig(os.path.join(out_path, f'{comb[1]}_vs_{comb[0]}.png'), bbox_inches='tight')
        if verbose:
            plt.show()
        plt.close()


def plot_lines(
                results: pd.DataFrame,
                samples: np.ndarray,
                metric: str,
                dataset: str,
                out_path: str,
                verbose: bool=True,
) -> None:
    plt.figure(figsize=(8, 4), dpi=100)
    ax = sns.lineplot(
                     data=results,
                     x="samples",
                     y="performance",
                     hue="training",
                     style="training",
                     ci=95,
                     err_style="band",
                     legend="full",
                     markers=True,
                     sort=True,
                     palette=['darkorange', 'darkslateblue',  'seagreen'],
    )
    
    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    ax.set_xticks(sorted(list(filter(lambda x: not np.isnan(x), results.samples.unique()))))
    ax.legend().set_title('')
    
    if metric == 'accuracy':
        ylabel = f'{metric.capitalize()} (%)'
    else:
        ylabel = metric.capitalize()
    
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10)
    ax.set_xlabel('Number of examples per class', fontsize=12, labelpad=7.5)
    ax.set_xticklabels(samples, fontsize=11)
    ax.legend(shadow=False, fancybox=True, loc='lower right')
    ax.set_title(dataset.upper())
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'results.png'), bbox_inches='tight')
    if verbose:
        plt.show()
    plt.close()