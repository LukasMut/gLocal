import argparse
from downstream.retrieval import embeddings
from downstream.retrieval import CLIP_MODELS
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--embeddings_dir')
parser.add_argument('--data_root', default='resources/flickr30k_images')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

os.makedirs(args.embeddings_dir, exist_ok=True)

for model_name in tqdm(CLIP_MODELS):
    embeddings.compute_embeddings(embeddings_folder=args.embeddings_dir, device=args.device,
                                  data_root=args.data_root, model_name=model_name)
