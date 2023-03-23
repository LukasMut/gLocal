import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from downstream.retrieval import CLIP_MODEL_MAPPING, CLIP_MODELS
from downstream.retrieval.eval import evaluate
from downstream.utils import THINGSFeatureTransform


def evaluate_normal_vs_transformed(
        embeddings_dir: str,
        data_root: str,
        transform_path: str,
        update_transforms: bool = False,
        concat_weight=None,
):
    all_results = []

    for model_name in tqdm(CLIP_MODELS, desc="CLIP model"):
        try:
            embeddings = np.load(os.path.join(embeddings_dir, f"{model_name}.npz"))
        except FileNotFoundError:
            warnings.warn(
                message=f"\nCould not find embedding file for {model_name}. Skipping current evaluation and continuing with next CLIP model...\n",
                category=UserWarning,
                stacklevel=1,
            )
            continue
        img_embedding = embeddings["images"]
        text_embedding = embeddings["text"]
        if update_transforms:
            model_key = CLIP_MODEL_MAPPING[model_name]
            things_feature_transform = THINGSFeatureTransform(
                source="custom",
                model_name=model_key,
                module="penultimate",
                path_to_transform=transform_path,
            )
            image_transformed = torch.tensor(
                things_feature_transform.transform_features(img_embedding)
            )
            text_transformed = torch.tensor(
                things_feature_transform.transform_features(text_embedding)
            )
            np.savez(
                os.path.join(embeddings_dir, f"{model_name}.npz"),
                images=img_embedding,
                text=text_embedding,
                image_transformed=image_transformed,
                text_transformed=text_transformed,
            )
        else:
            image_transformed = torch.from_numpy(embeddings["image_transformed"])
            text_transformed = torch.from_numpy(embeddings["text_transformed"])

        # Evaluate without transforms
        img_embedding = torch.from_numpy(img_embedding)
        text_embedding = torch.from_numpy(text_embedding)
        results = evaluate(img_embedding, text_embedding, dataset_root=data_root)
        results["model"] = model_name
        results["transform"] = False
        all_results.append(results)

        if concat_weight is not None:
            print("\nUsing weighted concat with", concat_weight)
            print(f"Shape: {image_transformed.shape}\n")
            image_transformed = torch.cat(
                img_embedding * (1 - concat_weight),
                image_transformed * concat_weight,
                dim=1,
            )
            text_transformed = torch.cat(
                text_embedding * (1 - concat_weight),
                text_transformed * concat_weight,
                dim=1,
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
        help="Off by default. Set to a weighing factor to concat embeddings and weigh by the <factor>",
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
