<div align="center">
    <a href="https://github.com/LukasMut/human_alignment/blob/main" rel="nofollow">
        <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9-blue.svg" alt="PyPI" />
    </a>
    <a href="https://github.com/psf/black" rel="nofollow">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
    </a>
</div>

# Improving neural network representations using human similarity judgments

## Environment setup and dependencies

We recommend to create a virtual environment (e.g., human_alignment), including all dependencies, via `conda`

```bash
$ conda env create --prefix /path/to/conda/envs/human_alignment --file envs/environment.yml
$ conda activate human_alignment
$ pip install git+https://github.com/openai/CLIP.git
```

Alternatively, dependencies can be installed via `pip`,

```bash
$ conda create --name human_alignment python=3.9
$ conda activate human_alignment
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install git+https://github.com/openai/CLIP.git
```

## Repository structure

```bash
root
├── envs
├── └── environment.yml
├── data
├── ├── __init__.py
├── ├── cifar.py
├── └── things.py
├── utils
├── ├── __init__.py
├── ├── analyses/*.py
├── ├── evaluation/*.py
├── └── probing/*.py
├── models
├── ├── __init__.py
├── ├── custom_mode.py
├── └── utils.py
├── .gitignore
├── README.md
├── main_embedding_sim_eval.py
├── main_embedding_triplet_eval.py
├── main_model_comparison.py
├── main_model_sim_eval.py
├── main_model_triplet_eval.py
├── main_probing.py
├── requirements.txt
├── search_temp_scaling.py
├── show_triplets.py
└── visualize_embeddings.py
```

## Usage

Run evaluation script on things triplet odd-one-out task with some pretrained model.

```python
$ python main_model_triplet_eval.py --data_root /path/to/data/name \ 
--dataset name \
--model_names resnet101 vgg11 clip_ViT-B/32 clip_RN50 vit_b_16 \
--module logits \
--overall_source thingsvision \
--sources torchvision torchvision custom custom torchvision  \
--model_dict_path /path/to/model_dict.json \
--batch_size 128 \
--distance cosine \
--out_path /path/to/results \
--device cpu \
--verbose \
--rnd_seed 42
```

Run evaluation script on multi-arrangement similarity judgements with some pretrained model.

```python
$ python main_model_sim_eval.py --data_root /path/to/data/name \ 
--dataset name \
--model_names resnet101 vgg11 clip_ViT-B/32 clip_RN50 vit_b_16 \
--module logits \
--overall_source thingsvision \
--sources torchvision torchvision custom custom torchvision  \
--model_dict_path /path/to/model_dict.json \
--batch_size 118 
--out_path /path/to/results \
--device cpu \
--verbose \
--rnd_seed 42 \
```


## Downstream Task Evaluations
We evaluate the transformation matrix obtained by probing on the Things task on various downstream tasks.

### CLIP Retrieval
We evaluate text -> image retrieval on the Flickr30K dataset. 
To compute the embeddings for all CLIP models, run:

```bash
python main_retrieval_init.py --embeddings_dir /home/space/datasets/things/downstream/clip-retrieval/retrieval_embeddings \
                              --data_root /home/space/datasets/things/downstream/clip-retrieval/flickr30k_images
```
(The embeddings are already computed on the TU cluster, so no need to run this step when working on the TU Cluster)

To evaluate the embeddings with and without transforms:

```bash
python main_retrieval_eval.py --out retrieval_results.csv \
                              --update_transforms \
                              --embeddings_dir /home/space/datasets/things/downstream/clip-retrieval/retrieval_embeddings \
                              --data_root /home/space/datasets/things/downstream/clip-retrieval/flickr30k_images
```
`--concat_weight` can be used to concat the transformed and normal embeddings and weigh the transformed ones.
`--transform_path` can be used to change the path from which the transformation matrices are loaded.









