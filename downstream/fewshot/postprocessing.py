import argparse
import os
import numpy as np
import pandas as pd
from typing import List, Optional, Union

GLOCAL_GRID = {
    "optim": ["sgd"],
    "eta": [0.01, 0.001, 0.0001],
    "lmbda": [10.0, 1.0, 0.1],
    "alpha": [0.5, 0.25, 0.1],
    "tau": [1.0, 0.5],
    "contrastive_batch_size": [1024],
}

GLOBAL_GRID = {
    "optim": ["adam", None],
    "eta": [0.0001, 1e-05],
    "lmbda": [100.0, 10.0, 1.0, 0.1, 0.001, 0.0001],
    "alpha": [None],
    "tau": [None],
    "contrastive_batch_size": [None],
}
NAIVE_GRID = {
    "optim": ["adam"],
    "eta": [0],
    "lmbda": [0],
    "alpha": [None],
    "tau": [None],
    "contrastive_batch_size": [None],
}
NAIVEBIAS_GRID = {
    "optim": ["adam"],
    "eta": [0],
    "lmbda": [-1],
    "alpha": [None],
    "tau": [None],
    "contrastive_batch_size": [None],
}
WITHOUT_GRID = {
    "optim": ["adam"],
    "eta": [0],
    "lmbda": [-2],
    "alpha": [None],
    "tau": [None],
    "contrastive_batch_size": [None],
}

MODEL2PRETTY = {
    "alexnet": "AlexNet",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vgg16": "VGG-16",
    "clip_RN50": "CLIP-RN50",
    "clip_ViT-L/14": "CLIP-ViT-L/14 (WIT)",
    "OpenCLIP_ViT-L-14_laion400m_e32": "CLIP-ViT-L/14 (LAION-400M)",
    "OpenCLIP_ViT-L-14_laion2b_s32b_b82k": "CLIP-ViT-L/14 (LAION-2B)",
}
MODEL2SRC = {
    "alexnet": "torchvision",
    "resnet18": "torchvision",
    "resnet50": "torchvision",
    "vgg16": "torchvision",
    "clip_RN50": "custom",
    "clip_ViT-L/14": "custom",
    "OpenCLIP_ViT-L-14_laion400m_e32": "custom",
    "OpenCLIP_ViT-L-14_laion2b_s32b_b82k": "custom",
}
PRETTY2ORDER = {
    MODEL2PRETTY["alexnet"]: 0,
    MODEL2PRETTY["vgg16"]: 1,
    MODEL2PRETTY["resnet18"]: 2,
    MODEL2PRETTY["resnet50"]: 3,
    MODEL2PRETTY["clip_RN50"]: 4,
    MODEL2PRETTY["clip_ViT-L/14"]: 5,
    MODEL2PRETTY["OpenCLIP_ViT-L-14_laion400m_e32"]: 6,
    MODEL2PRETTY["OpenCLIP_ViT-L-14_laion2b_s32b_b82k"]: 7,
    "baseline": 8,
}


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--results_root", type=str, help="path/to/results")
    aa("--out_dir", type=str, help="directory to save the processed results to")
    aa("--all_reps", action="store_true", help="whether to process all repetitions")
    args = parser.parse_args()
    return args


def get_file_paths(root: str):
    """Returns a list of paths to all .pkl files in the given root directory."""
    file_paths = []
    for root, dirs, files in os.walk(root):
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.splitext(file_path)[1] == ".pkl":
                file_paths += [file_path]
    return file_paths


def get_configs_data(data: pd.DataFrame):
    """Returns a list of dataframes, each containing the results for a unique configuration."""
    params = ["optim", "eta", "lmbda", "alpha", "tau", "contrastive_batch_size"]
    cfgs = list(set([tuple([d[p] for p in params]) for _, d in data.copy().iterrows()]))
    to_return = []
    for cfg in cfgs:
        sub_data = data
        for p in params:
            cfg_p = cfg[params.index(p)]
            if cfg_p is None or (type(cfg_p) != str and np.isnan(cfg_p)):
                sub_data = sub_data[sub_data[p].isna()]
            else:
                sub_data = sub_data[sub_data[p] == cfg_p]
        if len(sub_data) > 0:
            to_return.append(sub_data)
    return to_return


def get_fs_data(
    results_roots: Union[str, List[str]],
    dataset: str,
    task: str,
    models: List[str],
    srcs: List[str],
    module: str,
    sps: Optional[bool] = None,
    transform: str = "glocal",
):
    data = []
    n_files = 0
    if type(results_roots) == str:
        results_roots = [results_roots]

    # Load data files
    for base_folder in results_roots:
        print("Checking", base_folder)
        for m_i, model in enumerate(models):
            path = os.path.join(
                base_folder,
                dataset + (f"_{task}" if task else ""),
                srcs[m_i],
                model,
                module,
            )

            for file in get_file_paths(path):
                if sps is not None:
                    if sps and "True" not in file:
                        continue
                    elif not sps and "False" not in file:
                        continue

                data_sub = pd.read_pickle(file)
                # Filtering for hyper-param grid
                if transform == "glocal":
                    allowed = GLOCAL_GRID
                elif transform == "global":
                    # global
                    allowed = GLOBAL_GRID
                elif transform == "naive+bias":
                    allowed = NAIVEBIAS_GRID
                else:
                    allowed = NAIVE_GRID
                if all([data_sub[p].iloc[0] in allowed[p] for p in allowed.keys()]) or \
                    all([data_sub[p].iloc[0] in WITHOUT_GRID[p] for p in WITHOUT_GRID.keys()]):
                    del data_sub["classes"]
                    data.append(data_sub)
                    n_files += 1

    # Postprocess data content
    if len(data) > 0:
        data = pd.concat(data).drop_duplicates()
        for column in ["eta", "lmbda"]:
            data[column] = data[column].astype(float)
        data["optim"] = data["optim"].astype(str)
        if transform == "glocal":
            for column in ["tau", "contrastive_batch_size", "alpha"]:
                data[column] = data[column].astype(float)
        data["transform_type"] = transform
        data["model_pretty"] = data["model"].replace(MODEL2PRETTY)
    else:
        data = None
    print(f"{dataset}/{task} Files found: {n_files}")
    return data


def preprocess_reps(data: pd.DataFrame):
    configs_data = get_configs_data(data)
    shots = list(set(data.n_train))
    regressors = list(set(data.regressor))
    models = list(set(data.model))
    spss = list(set(data.samples_per_superclass))
    reps = list(set(data.repetition))

    new_cols = ["fs_accuracy_t"]
    rows = []
    for sps in spss:
        for s_i, s in enumerate(shots):
            for r_i, r in enumerate(regressors):
                for m_i, model in enumerate(models):
                    # Get basline accuracy
                    baseline = [None for _ in range(len(reps))]
                    for cd in configs_data:
                        selection = cd[
                            (cd["model"] == model)
                            & (cd["regressor"] == r)
                            & (cd["n_train"] == s)
                            & (cd["samples_per_superclass"] == sps)
                            & (cd["transform"] == False)
                        ]
                        if len(selection) == len(reps):
                            for rep_i, rep in enumerate(reps):
                                baseline[rep_i] = np.sum(
                                    selection[(selection["repetition"] == rep)][
                                        "accuracy"
                                    ]
                                )
                            break
                    if any([b is None for b in baseline]):
                        # Skip if no baseline available
                        continue

                    # Get transformed accuracy
                    for cd in configs_data:
                        for rep_i, rep in enumerate(reps):
                            selection = cd[
                                (cd["model"] == model)
                                & (cd["regressor"] == r)
                                & (cd["n_train"] == s)
                                & (cd["samples_per_superclass"] == sps)
                                & (cd["repetition"] == rep)
                            ]
                            acc = baseline[rep_i]
                            acc_t = np.sum(
                                selection[(selection["transform"] == True)]["accuracy"]
                            )

                            if acc_t is None or len(selection["model"]) == 0:
                                continue

                            assert acc <= 1 and acc_t <= 1

                            row = [
                                (acc if k == "accuracy" else selection[k].iloc[0])
                                for k in selection.columns
                            ] + [acc_t]
                            rows.append(row)
    new_data = pd.DataFrame(rows, columns=[c for c in data.columns] + new_cols)
    new_data["fs_accuracy"] = new_data["accuracy"].astype(float)
    del new_data["accuracy"]
    for col in new_cols:
        new_data[col] = new_data[col].astype(float)
    return new_data


def avg_reps(data: pd.DataFrame):
    """Returns a dataframe with the mean of the given data across repetitions."""
    try:
        configs_data = get_configs_data(data)
    except:
        return None
    shots = list(set(data.n_train))
    regressors = list(set(data.regressor))
    models = list(set(data.model))
    spss = list(set(data.samples_per_superclass))
    reps = list(set(data.repetition))

    new_cols = ["fs_accuracy_t", "fs_accuracy_diff_mean", "fs_accuracy_diff_std"]
    rows = []
    for sps in spss:
        for s_i, s in enumerate(shots):
            for r_i, r in enumerate(regressors):
                for m_i, model in enumerate(models):
                    # Get basline accuracy
                    baseline = None
                    baseline_reps = None
                    for cd in configs_data:
                        selection = cd[
                            (cd["model"] == model)
                            & (cd["regressor"] == r)
                            & (cd["n_train"] == s)
                            & (cd["samples_per_superclass"] == sps)
                        ]
                        if len(selection[(selection["transform"] == False)]) > 0:
                            baseline = np.mean(
                                selection[(selection["transform"] == False)]["accuracy"]
                            )
                            baseline_reps = [
                                float(
                                    selection[
                                        (selection["transform"] == False)
                                        & (selection["repetition"] == rep)
                                    ]["accuracy"]
                                )
                                for rep in reps
                            ]
                            break

                    if baseline is None:
                        print(
                            "Warning: Baseline is none",
                            model,
                            data.dataset.unique(),
                            "shots =",
                            s,
                        )
                        continue
                    for cd in configs_data:
                        selection = cd[
                            (cd["model"] == model)
                            & (cd["regressor"] == r)
                            & (cd["n_train"] == s)
                            & (cd["samples_per_superclass"] == sps)
                        ]
                        acc = baseline
                        acc_t = np.mean(
                            selection[(selection["transform"] == True)]["accuracy"]
                        )

                        if baseline is not None:
                            try:
                                diff = [
                                    float(
                                        selection[
                                            (selection["transform"] == False)
                                            & (selection["repetition"] == rep)
                                        ]["accuracy"]
                                    )
                                    - baseline_reps[i_rep]
                                    for i_rep, rep in enumerate(sorted(reps))
                                ]
                                acc_diff_mean = np.mean(diff)
                                acc_diff_std = np.std(diff)
                            except:
                                acc_diff_mean = None
                                acc_diff_std = None

                        if acc_t is None or pd.isna(acc_t) or len(selection["model"]) == 0:
                            continue

                        row = [
                            (acc if k == "accuracy" else selection[k].iloc[0])
                            for k in selection.columns
                        ] + [acc_t, acc_diff_mean, acc_diff_std]
                        rows.append(row)
    new_data = pd.DataFrame(rows, columns=[c for c in data.columns] + new_cols)
    new_data["fs_accuracy"] = new_data["accuracy"].astype(float)
    for col in new_cols:
        new_data[col] = new_data[col].astype(float)
    del new_data["accuracy"]
    del new_data["repetition"]
    return new_data


def main(args):
    models = [k for k in MODEL2PRETTY.keys()]
    srcs = [MODEL2SRC[m] for m in models]
    module = "penultimate"

    datasets_fs = ["imagenet", "imagenet", "cifar100", "cifar100", "SUN397", "DTD"]
    tasks_fs = ["entity13", "entity30", None, "coarse", None, None]
    task_fs_is_super = [True, True, False, True, False, False]

    print("### Loading FS naive data")
    datas_naive_fs = [
        get_fs_data(args.results_root, dataset, task, models, srcs, module, sps, transform="naive")
        for dataset, task, sps in zip(datasets_fs, tasks_fs, task_fs_is_super)
    ]
    print("### Loading FS naive+bias data")
    datas_naiveb_fs = [
        get_fs_data(args.results_root, dataset, task, models, srcs, module, sps, transform="naive+bias")
        for dataset, task, sps in zip(datasets_fs, tasks_fs, task_fs_is_super)
    ]
    print("### Loading FS global data")
    datas_global_fs = [
        get_fs_data(args.results_root, dataset, task, models, srcs, module, sps, transform="global")
        for dataset, task, sps in zip(datasets_fs, tasks_fs, task_fs_is_super)
    ]
    print("### Loading FS glocal data")
    datas_fs = [
        get_fs_data(args.results_root, dataset, task, models, srcs, module, sps)
        for dataset, task, sps in zip(datasets_fs, tasks_fs, task_fs_is_super)
    ]

    # Make task names pretty
    tasks_fs = [
        ((d + ("-" if t else "")) if d != "imagenet" else "") + (t if t else "")
        for d, t in zip(datasets_fs, tasks_fs)
    ]

    # Filter out tasks that for which no results are available yet
    tasks_fs = [ds for d, ds in zip(datas_fs, tasks_fs) if d is not None]
    datas_fs = [data for data in datas_fs if data is not None]
    datas_global_fs = [data for data in datas_global_fs if data is not None]

    for dt in zip(datas_fs, datas_global_fs, datas_naive_fs, datas_naiveb_fs, tasks_fs):
        for d in dt[:-1]:
            if d is not None:
                d["dataset"] = dt[-1]

    # Create output directory
    if not os.path.exists(args.out_dir):
        print("\nOutput directory does not exist...")
        print("Creating output directory to save results...\n")
        os.makedirs(args.out_dir)

    print("### Averaging FS data")
    data_avgs_naive_fs = [avg_reps(data) for data in datas_naive_fs]
    data_avgs_naiveb_fs = [avg_reps(data) for data in datas_naiveb_fs]
    data_avgs_fs = [avg_reps(data) for data in datas_fs]
    data_avgs_global_fs = [avg_reps(data) for data in datas_global_fs]

    # Save
    pd.concat(data_avgs_fs).to_pickle(os.path.join(args.out_dir, "fs_results_glocal.pkl"))
    pd.concat(data_avgs_global_fs).to_pickle(os.path.join(args.out_dir, "fs_results_global.pkl"))
    pd.concat(data_avgs_naive_fs).to_pickle(os.path.join(args.out_dir, "fs_results_naive.pkl"))
    pd.concat(data_avgs_naiveb_fs).to_pickle(os.path.join(args.out_dir, "fs_results_naive+bias.pkl"))

    if args.all_reps:
        print("### Postprocessing not-averaged FS data")
        datas_fs = [preprocess_reps(data) for data in datas_fs]
        datas_global_fs = [preprocess_reps(data) for data in datas_global_fs]

        # Save
        pd.concat(datas_fs).to_pickle(os.path.join(args.out_dir, "fs_results_glocal_large.pkl"))
        pd.concat(datas_global_fs).to_pickle(os.path.join(args.out_dir, "fs_results_global_large.pkl"))


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    main(args)
