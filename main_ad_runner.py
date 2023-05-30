from main_anomaly_detection import main
import os
from os.path import join
import numpy as np
import argparse
from tqdm import tqdm
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

model_mapping = {
    'clip_ViT-L/14': 'clip_ViT-L-14',
    'clip_RN50': 'clip_RN50',
    'OpenCLIP_ViT-L-14_laion2b_s32b_b82k': 'OpenCLIP_ViT-L-14_laion2b_s32b_b82k',
    'OpenCLIP_ViT-L-14_laion400m_e32': 'OpenCLIP_ViT-L-14_laion400m_e32',
    'vgg16': 'vgg16',
    'alexnet': 'alexnet',
    'resnet18': 'resnet18',
    'resnet50': 'resnet50',
}

transform_paths = {
    'global': '/home/space/datasets/things/transforms/globals',
    'glocal': '/home/space/datasets/things/probing/results/',
    'naive': '/home/space/datasets/things/transforms/naive_transforms.pkl'
}

module_mapping = {
    'vgg16': 'classifier.3',
    'alexnet': 'classifier.4',
    'resnet18': 'avgpool',
    'resnet50': 'avgpool'
}

dataset_classes = {'cifar10': 10, 'cifar100': 100,
                   'cifar100-coarse': 20, 'cifar100-shift': 20,
                   'cifar10-rvo': 10,
                   'cifar100-rvo': 100,
                   'cifar100-coarse-rvo': 20, 'cifar10vs100': 1, 'dtd': 47,
                   'flowers': 102,
                   'dtd-rvo': 47,
                   'flowers-rvo': 102,
                   'imagenet30': 30,
                   'imagenet30-rvo': 30}

breeds_classes = [17, 13, 30, 26]
for i, task in enumerate(("living17", "entity13", "entity30", "nonliving26")):
    dataset_classes[f'breeds-{task}'] = breeds_classes[i]
    dataset_classes[f'breeds-{task}-rvo'] = breeds_classes[i]

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="/home/spaces/datasets")
parser.add_argument("--models", nargs='+', default=None)
parser.add_argument("--datasets", nargs='+', default=None)
parser.add_argument('--device', default='cuda')
parser.add_argument('--clip-zero-shot', action='store_true')
parser.add_argument('--shift-indices', type=int, nargs='+', default=[0, 1, 2])
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--type', choices=transform_paths.keys())
parser.add_argument('--results-root', default='ad-results')
args = parser.parse_args()

if args.datasets is None:
    datasets = dataset_classes.keys()
    print(datasets)
else:
    datasets = args.datasets

if args.models is None:
    models = model_mapping.keys()
else:
    models = args.models

for model in tqdm(models):
    if model in ['resnet18', 'resnet50', 'vgg16', 'alexnet']:
        source = 'torchvision'
    else:
        source = 'custom'

    module_type = 'penultimate'
    if model in module_mapping:
        module = module_mapping[model]
    else:
        module = 'visual'

    for dataset in tqdm(datasets):
        # create results path
        result_dir = join(args.results_root, dataset, model_mapping[model])
        result_path = join(result_dir, f'{args.type}_results.json')
        os.makedirs(result_dir, exist_ok=True)

        if source == 'torchvision' and dataset.startswith('imagenet30'):
            # imagenet pretrained doesn't make sense here
            continue

        if args.type == 'naive':
            transforms = [transform_paths[args.type]]
        else:
            # get all transform paths
            transform_path = join(transform_paths[args.type], source, model)
            transforms = []
            for root, dirs, files in os.walk(transform_path):
                for filename in files:
                    filename = os.path.join(root, filename)
                    if not filename.endswith('.npz'):
                        continue
                    if args.type == 'glocal':
                        data = np.load(filename)
                        if np.sum(np.isnan(data['weights'])) > 0:
                            continue
                    transforms.append(filename)

        # run eval
        ad_results = main(dataset=dataset, data_root=args.data_root,
                          source=source,
                          model_name=model, module=module,
                          module_type=module_type,
                          archive_path=None,
                          device=args.device,
                          output_file=result_path,
                          path_to_transforms=transforms,
                          num_classes=dataset_classes[dataset],
                          shift_indices=args.shift_indices,
                          knn_k=args.k,
                          clip_zero_shot=args.clip_zero_shot)
