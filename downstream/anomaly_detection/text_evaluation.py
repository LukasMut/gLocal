import numpy as np
from .cifar import ADCIFAR10, ADCIFAR100, ADCIFAR100Shift, ADCIFAR100Coarse
import torch
from sklearn.metrics import roc_auc_score
from .dtd import ADDTD
from .fine_grained import ADFlowers
from thingsvision import get_extractor
import torch.nn.functional as F
import os
import open_clip

Array = np.array


class ImageIterator:

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        for x, _ in self.data_loader:
            yield x

    def __len__(self):
        return len(self.data_loader)


class ADZeroShotEvaluator:

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
            dataset.replace('-rvo', '')

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

        model, _, preprocess = open_clip.create_model_and_transforms(model_params['variant'],
                                                                     pretrained='openai')
        model = model.to(device)

        self.normal_anchors = []
        self.anomaly_anchors = []

        print('embedding text')
        for cls_name in anomaly_ds.class_names():
            with torch.no_grad():
                normal_text = open_clip.tokenize(['a photo of a ' + cls_name]).to(device)
                normal_text_features = model.encode_text(normal_text)
                anomaly_text = open_clip.tokenize(['a photo of something else']).to(device)
                anomaly_text_features = model.encode_text(anomaly_text)
            normal_anchor = F.normalize(normal_text_features.mean(dim=0), dim=-1)
            anomaly_anchor = F.normalize(anomaly_text_features.mean(dim=0), dim=-1)

            print("computed anchors", normal_anchor.shape, anomaly_anchor.shape)
            self.normal_anchors.append(normal_anchor)
            self.anomaly_anchors.append(anomaly_anchor)

        self.normal_anchors = torch.stack(self.normal_anchors)
        self.anomaly_anchors = torch.stack(self.anomaly_anchors)

    def evaluate(self, normal_classes, things_transform=None, *args, **kwargs):

        test_features = self.test_features

        normal_anchors = self.normal_anchors
        anomaly_anchors = self.anomaly_anchors

        if things_transform is not None:
            test_features = things_transform.transform_features(test_features)
            normal_anchors = things_transform.transform_features(normal_anchors)
            anomaly_anchors = things_transform.transform_features(anomaly_anchors)

        test_features = F.normalize(test_features, dim=-1)

        aucs = []
        for normal_cls in normal_classes:
            test_reduced, test_reduced_labels = self.dataset.reduce_test(test_embeddings=test_features,
                                                                         cls=normal_cls)

            text_query = torch.stack([normal_anchors[normal_cls], anomaly_anchors[normal_cls]])
            sim = F.softmax(torch.mm(test_reduced, text_query.t()), dim=-1)
            anomaly_scores = (sim[:, 1] - sim[:, 0]).cpu().numpy()
            aucs.append(roc_auc_score(test_reduced_labels, anomaly_scores))
        return aucs
