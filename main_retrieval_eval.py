import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from downstream.retrieval import CLIP_MODEL_MAPPING, CLIP_MODELS
from downstream.retrieval.eval import evaluate
from downstream.retrieval.transform import ThingsFeatureTransform


def evaluate_normal_vs_transformed(
    embeddings_dir,
    data_root,
    transform_path,
    update_transforms=False,
    concat_weight=None,
):
    all_results = []
    if update_transforms:
        things_feature_transform = ThingsFeatureTransform(transform_path=transform_path)
    for model_name in tqdm(CLIP_MODELS):
        embeddings = np.load(os.path.join(embeddings_dir, f"{model_name}.npz"))
        text = torch.tensor(embeddings["text"])
        image = torch.tensor(embeddings["images"])
        if update_transforms:
            transform_model_key = CLIP_MODEL_MAPPING[model_name]
            image_transformed = torch.tensor(
                things_feature_transform.transform_features(
                    embeddings["images"], model_name=transform_model_key
                )
            )
            text_transformed = torch.tensor(
                things_feature_transform.transform_features(
                    embeddings["text"], model_name=transform_model_key
                )
            )
            np.savez(
                os.path.join(embeddings_dir, f"{model_name}.npz"),
                images=embeddings["images"],
                text=embeddings["text"],
                image_transformed=image_transformed,
                text_transformed=text_transformed,
            )
        else:
            image_transformed = torch.tensor(embeddings["image_transformed"])
            text_transformed = torch.tensor(embeddings["text_transformed"])

        # Evaluate without transforms
        results = evaluate(image, text, dataset_root=data_root)
        results["model"] = model_name
        results["transform"] = False
        all_results.append(results)

        if concat_weight is not None:
            print("Using weighted concat with", concat_weight)
            print(image_transformed.shape)
            image_transformed = torch.cat(
                image * (1 - concat_weight), image_transformed * concat_weight, dim=1
            )
            text_transformed = torch.cat(
                text * (1 - concat_weight), text_transformed * concat_weight, dim=1
            )

        # Evaluate with transforms
        results_t = evaluate(
            image_transformed, text_transformed, dataset_root=data_root
        )
        results_t["model"] = model_name
        results_t["transform"] = True
        all_results.append(results_t)
    return pd.DataFrame(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir")
    parser.add_argument("--update_transforms", action="store_true")
    parser.add_argument(
        "--concat-weight",
        type=float,
        default=None,
        help="Off by default. Set to a weighing factor to concat embeddings and weigh by the "
        "factor",
    )
    parser.add_argument("--data_root", default="resources/flickr30k_images")
    parser.add_argument(
        "--transform_path",
        default="/home/space/datasets/things/transforms/transforms_without_norm.pkl",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    results_df = evaluate_normal_vs_transformed(
        embeddings_dir=args.embeddings_dir,
        update_transforms=args.update_transforms,
        concat_weight=args.concat_weight,
        data_root=args.data_root,
        transform_path=args.transform_path,
    )
    results_df.to_csv(args.out, index=False)
