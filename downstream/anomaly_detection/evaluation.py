import numpy as np
from .cifar import ADCIFAR10, ADCIFAR100, ADCIFAR100Shift
import torch
from sklearn.metrics import roc_auc_score
from .imagenet30 import ADImageNet
from .dtd import ADDTD
from .fine_grained import ADFlowers
from thingsvision import get_extractor
import torch.nn.functional as F

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

    def __init__(self, dataset, model_name, source, module, model_params, data_dir, normal_cls=0, device='cuda',
                 things_transform=None, **kwargs):

        self.extractor = get_extractor(
            model_name=model_name,
            source=source,
            device=device,
            pretrained=True,
            model_parameters=model_params
        )
        transform = self.extractor.get_transformations()
        self.device = device
        self.things_transform = things_transform
        self.module = module

        if dataset == 'cifar10':
            anomaly_ds = ADCIFAR10(transform=transform, data_dir=data_dir)
        elif dataset == 'cifar100':
            anomaly_ds = ADCIFAR100(normal_classes=[normal_cls], transform=transform, data_dir=data_dir)
        elif dataset == 'imagenet':
            anomaly_ds = ADImageNet(normal_classes=[normal_cls], transform=transform, data_dir=data_dir)
        elif dataset == 'dtd':
            anomaly_ds = ADDTD(normal_classes=[normal_cls], transform=transform, data_dir=data_dir)
        elif dataset == 'flowers':
            anomaly_ds = ADFlowers(normal_classes=[normal_cls], transform=transform, data_dir=data_dir)
        elif dataset == 'cifar100-shift':
            anomaly_ds = ADCIFAR100Shift(normal_class=normal_cls, transform=transform,
                                         data_dir=data_dir, **kwargs)
        else:
            raise ValueError()

        self.dataset = anomaly_ds
        self.dataset.setup()

    def evaluate(self, normal_classes, knn_k=5, do_transform=False):

        train_features = self.extractor.extract_features(
            batches=ImageIterator(self.dataset.train_dataloader()),
            module_name=self.module,
            flatten_acts=True
        )
        train_features = F.normalize(torch.tensor(train_features).to(torch.float32), dim=-1)

        test_features = self.extractor.extract_features(
            batches=ImageIterator(self.dataset.test_dataloader()),
            module_name=self.module,
            flatten_acts=True
        )
        test_features = F.normalize(torch.tensor(test_features).to(torch.float32), dim=-1)

        aucs = []
        for normal_cls in normal_classes:
            train_reduced = self.dataset.reduce_train(train_embeddings=train_features, normal_cls=normal_cls)
            test_reduced, test_reduced_labels = self.dataset.reduce_test(test_embeddings=test_features,
                                                                         normal_cls=normal_cls)

            print(train_reduced.shape)
            print(test_reduced.shape)
            sim_matrix = torch.mm(test_reduced, train_reduced.t())
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            anomaly_scores = - sim_weight.mean(dim=-1).cpu().numpy()
            aucs.append(roc_auc_score(test_reduced_labels, anomaly_scores))
        return aucs
