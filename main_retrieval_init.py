import argparse
from downstream.retrieval import embeddings
from downstream.retrieval import CLIP_MODELS
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--embedding-dir')
parser.add_argument('--data_root', default='resources/flickr30k_images')
args = parser.parse_args()

os.makedirs(args.embedding_dir)

for model_name in tqdm(CLIP_MODELS):
    embeddings.compute_embeddings(embeddings_folder=args.embedding_dir, device='cuda',
                                  data_root=args.data_root, model_name=model_name)
