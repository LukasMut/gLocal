import pandas as pd
import torch
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from collections import defaultdict


def evaluate(image_features, text_features, dataset_root="flickr30k_images", device='cuda'):
    image_features = image_features.to(device)
    text_features = text_features.to(device)

    df = pd.read_csv(os.path.join(dataset_root, 'results.csv'), sep="|")
    image_names = df.image_name.unique()

    test_images = []
    with open(os.path.join(dataset_root, 'test.txt'), "r") as f:
        for line in f:
            test_images.append(line.replace("\n", "") + ".jpg")
    image_mask = torch.zeros(len(image_names), dtype=torch.bool)
    for i, img in enumerate(image_names):
        if img in test_images:
            image_mask[i] = 1
    test_image_features = image_features[image_mask]
    image_names_test = image_names[image_mask]
    selector = df.image_name.isin(test_images)
    test = df[selector]

    total = 0
    correct = defaultdict(int)
    for idx, row in tqdm(test.iterrows()):
        image_name = row["image_name"]
        target = np.argmax(image_name == image_names_test).item()
        dist = F.cosine_similarity(test_image_features.float(), text_features[idx].unsqueeze(0))
        index_sorted = torch.argsort(dist)
        top_10 = list(reversed(index_sorted[-10:].tolist()))
        if target in top_10:
            correct["R@10"] += 1
        if target in top_10[:5]:
            correct["R@5"] += 1
        if top_10[0] == target:
            correct["R@1"] += 1
        total += 1
    result = {}
    for key in correct.keys():
        result[key] = correct[key] / total
    return result
