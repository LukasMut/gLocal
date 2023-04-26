import numpy as np
from .cifar import ADCIFAR10, ADCIFAR100, ADCIFAR100Shift, ADCIFAR100Coarse
import torch
from sklearn.metrics import roc_auc_score
from .dtd import ADDTD
from .fine_grained import ADFlowers
from thingsvision import get_extractor
import torch.nn.functional as F
import os

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

        if model_params is not None and "variant" in model_params:
            variant = model_params['variant']
            training_dataset = model_params['dataset']
            cache_path = f'{dataset}_{model_name}_{variant}_{training_dataset}_{module}.npz'
        else:
            cache_path = f'{dataset}_{model_name}__{module}.npz'

        cache_path = cache_path.replace('/', '-')
        self.cache_path = os.path.join(cache_dir, cache_path)

        if dataset == 'cifar10':
            anomaly_ds = ADCIFAR10(transform=transform, data_dir=data_dir)
        elif dataset == 'cifar100':
            anomaly_ds = ADCIFAR100(transform=transform, data_dir=data_dir)
        elif dataset == 'dtd':
            anomaly_ds = ADDTD(transform=transform, data_dir=data_dir)
        elif dataset == 'flowers':
            anomaly_ds = ADFlowers(transform=transform, data_dir=data_dir)
        elif dataset == 'cifar100-shift':
            anomaly_ds = ADCIFAR100Shift(transform=transform, data_dir=data_dir, **kwargs)
        elif dataset == 'cifar100-coarse':
            anomaly_ds = ADCIFAR100Coarse(transform=transform, data_dir=data_dir)
        else:
            raise ValueError()

        self.dataset = anomaly_ds
        self.dataset.setup()

        if os.path.exists(self.cache_path):
            saved_features = np.load(self.cache_path)
            self.train_features = saved_features['train']
            self.test_features = saved_features['test']
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
            np.savez(self.cache_path, train=self.train_features, test=self.test_features)

        self.train_features = torch.tensor(self.train_features).to(torch.float32).to(self.device)
        self.test_features = torch.tensor(self.test_features).to(torch.float32).to(self.device)

    def evaluate(self, normal_classes, knn_k=5, things_transform=None):

        train_features = self.train_features
        test_features = self.test_features

        if things_transform is not None:
            train_features = things_transform.transform_features(train_features)
        train_features = F.normalize(train_features, dim=-1)

        if things_transform is not None:
            test_features = things_transform.transform_features(test_features)
        test_features = F.normalize(test_features, dim=-1)

        aucs = []
        for normal_cls in normal_classes:
            train_reduced = self.dataset.reduce_train(train_embeddings=train_features, normal_cls=normal_cls)
            test_reduced, test_reduced_labels = self.dataset.reduce_test(test_embeddings=test_features,
                                                                         normal_cls=normal_cls)

            sim_matrix = torch.mm(test_reduced, train_reduced.t())
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            anomaly_scores = - sim_weight.mean(dim=-1).cpu().numpy()
            aucs.append(roc_auc_score(test_reduced_labels, anomaly_scores))
        return aucs
