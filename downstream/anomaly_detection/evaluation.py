import numpy as np
from .cifar import ADCIFAR10, ADCIFAR100, ADCIFAR100Shift, ADCIFAR100Coarse, ADCIFAR10vs100
import torch
from sklearn.metrics import roc_auc_score
from .dtd import ADDTD
from .fine_grained import ADFlowers, ADCUB2011
from .imagenet import ADImageNet30, ADBreeds
from thingsvision import get_extractor
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

Array = np.array


class ImageIterator:

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        for x, _ in self.data_loader:
            yield x

    def __len__(self):
        return len(self.data_loader)


class ADEvaluator:

    def __init__(self, dataset, model_name, source, module, model_params, data_dir,
                 device='cuda', cache_dir='cache', **kwargs):

        self.extractor = get_extractor(
            model_name=model_name,
            source=source,
            device=device,
            pretrained=True,
            model_parameters=model_params
        )
        transform = self.extractor.get_transformations()
        self.device = device
        self.module = module

        mode = 'ovr'
        if dataset.endswith('-rvo'):
            mode = 'rvo'
            dataset = dataset.replace('-rvo', '')

        def get_embeddings_root():
            base = '/home/space/datasets/imagenet_extracted/2012/features/'
            name = model_name
            if model_params is not None and "variant" in model_params:
                variant = model_params['variant']
                if 'dataset' in model_params:
                    training_dataset = model_params['dataset']
                    name = f'{model_name}_{variant}_{training_dataset}'
                else:
                    name = f'{model_name}_{variant}'
            return os.path.join(base, source, name, 'penultimate')

        if dataset == 'cifar10':
            anomaly_ds = ADCIFAR10(transform=transform, data_dir=data_dir, mode=mode)
        elif dataset == 'cifar100':
            anomaly_ds = ADCIFAR100(transform=transform, data_dir=data_dir, mode=mode)
        elif dataset == 'dtd':
            anomaly_ds = ADDTD(transform=transform, data_dir=data_dir, mode=mode)
        elif dataset == 'flowers':
            anomaly_ds = ADFlowers(transform=transform, data_dir=data_dir, mode=mode)
        elif dataset == 'cifar100-shift':
            anomaly_ds = ADCIFAR100Shift(transform=transform, data_dir=data_dir, mode=mode, **kwargs)
        elif dataset == 'cifar100-coarse':
            anomaly_ds = ADCIFAR100Coarse(transform=transform, data_dir=data_dir, mode=mode)
        elif dataset == 'cifar10vs100':
            anomaly_ds = ADCIFAR10vs100(transform=transform, data_dir=data_dir, mode=mode)
        elif dataset == 'cub2011':
            anomaly_ds = ADCUB2011(transform=transform, data_dir=data_dir, mode=mode)
        elif dataset == 'imagenet30':
            anomaly_ds = ADImageNet30(transform=transform, mode=mode, embeddings_root=get_embeddings_root())
        elif dataset.startswith('breeds-'):
            tokens = dataset.split('-')
            assert len(tokens) == 2 and tokens[0] == 'breeds'
            anomaly_ds = ADBreeds(transform=transform, mode=mode,
                                  task_name=tokens[1], embeddings_root=get_embeddings_root())
        else:
            raise ValueError()

        self.dataset = anomaly_ds
        self.dataset.setup()

        dataset_key = self.dataset.cache_name()
        if model_params is not None and "variant" in model_params:
            variant = model_params['variant']
            if 'dataset' in model_params:
                training_dataset = model_params['dataset']
                cache_path = f'{dataset_key}_{model_name}_{variant}_{training_dataset}_{module}.npz'
            else:
                cache_path = f'{dataset_key}_{model_name}_{variant}_{module}.npz'
        else:
            cache_path = f'{dataset_key}_{model_name}__{module}.npz'
        cache_path = cache_path.replace('/', '-')
        self.cache_path = os.path.join(cache_dir, cache_path)

        if os.path.exists(self.cache_path):
            saved_features = np.load(self.cache_path)
            self.train_features = saved_features['train']
            self.test_features = saved_features['test']
        else:
            if self.dataset.is_embedded():
                def extract_features(ds):
                    dl = DataLoader(ds, batch_size=128, num_workers=8)
                    features = []
                    for x, y in dl:
                        features.append(x)
                    features = torch.cat(features, dim=0)
                    return features

                self.train_features = extract_features(self.dataset._train)
                self.test_features = extract_features(self.dataset._test)
            else:
                self.train_features = self.extractor.extract_features(
                    batches=ImageIterator(self.dataset.train_dataloader()),
                    module_name=self.module,
                    flatten_acts=True
                )
                self.test_features = self.extractor.extract_features(
                    batches=ImageIterator(self.dataset.test_dataloader()),
                    module_name=self.module,
                    flatten_acts=True
                )
            if not self.dataset.is_embedded():
                np.savez(self.cache_path, train=self.train_features, test=self.test_features)

        self.train_features = torch.tensor(self.train_features).to(torch.float32).to(self.device)
        self.test_features = torch.tensor(self.test_features).to(torch.float32).to(self.device)

    def evaluate(self, normal_classes, knn_k=5, things_transform=None):
        train_features = self.train_features.clone()
        test_features = self.test_features.clone()

        def transform_features(embeddings):
            return things_transform.transform_features(embeddings)

        if things_transform is not None:
            train_features = transform_features(train_features)
        train_features = F.normalize(train_features, dim=-1)

        if things_transform is not None:
            test_features = transform_features(test_features)
        test_features = F.normalize(test_features, dim=-1)

        aucs = []
        for cls in normal_classes:
            train_reduced = self.dataset.reduce_train(train_embeddings=train_features, cls=cls)
            test_reduced, test_reduced_labels = self.dataset.reduce_test(test_embeddings=test_features,
                                                                         cls=cls)
            sim_matrix = torch.mm(test_reduced, train_reduced.t())
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            anomaly_scores = - sim_weight.mean(dim=-1).cpu().numpy()
            aucs.append(roc_auc_score(test_reduced_labels, anomaly_scores))
        return aucs
