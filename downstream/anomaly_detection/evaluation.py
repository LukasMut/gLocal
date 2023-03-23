import numpy as np
from .cifar import ADCIFAR10, ADCIFAR100, ADCIFAR100Shift
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from .imagenet30 import ADImageNet
from tqdm.auto import tqdm
from .dtd import ADDTD
from .fine_grained import ADFlowers

Array = np.array


class ADEvaluator:

    def __init__(self, dataset, model, data_dir, normal_cls=0, device='cuda', things_transform=None, **kwargs):
        imagenet_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        ])
        cifar_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(*[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        ])

        if dataset == 'cifar10':
            anomaly_ds = ADCIFAR10(normal_classes=[normal_cls], transform=cifar_transform, data_dir=data_dir)
        elif dataset == 'cifar100':
            anomaly_ds = ADCIFAR100(normal_classes=[normal_cls], transform=cifar_transform, data_dir=data_dir)
        elif dataset == 'imagenet':
            anomaly_ds = ADImageNet(normal_classes=[normal_cls], transform=imagenet_transform, data_dir=data_dir)
        elif dataset == 'dtd':
            anomaly_ds = ADDTD(normal_classes=[normal_cls], transform=imagenet_transform, data_dir=data_dir)
        elif dataset == 'flowers':
            anomaly_ds = ADFlowers(normal_classes=[normal_cls], transform=imagenet_transform, data_dir=data_dir)
        elif dataset == 'cifar100-shift':
            anomaly_ds = ADCIFAR100Shift(normal_class=normal_cls, transform=cifar_transform,
                                         data_dir=data_dir, **kwargs)
        else:
            raise ValueError()
        self.model = model
        self.model.eval()
        self.dataset = anomaly_ds
        self.dataset.setup()
        self.device = device
        self.model = model.to(device)
        self.things_transform = things_transform

    def evaluate(self, knn_k=5, do_transform=False):
        train_features = ()
        with torch.no_grad():
            for x, _ in tqdm(self.dataset.train_dataloader()):
                x = x.to(self.device)
                feats = self.model(x)
                if do_transform:
                    feats = self.things_transform.transform_features(feats)
                feats = F.normalize(feats, dim=1)
                train_features += (feats,)
        train_features = torch.cat(train_features).t().contiguous()

        test_anomaly_scores = []
        test_labels = []
        with torch.no_grad():
            for x, y in tqdm(self.dataset.test_dataloader()):
                x = x.to(self.device)
                feats = self.model(x)
                if do_transform:
                    feats = self.things_transform.transform_features(feats)
                feats = F.normalize(feats, dim=1)
                sim_matrix = torch.mm(feats, train_features)
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                anomaly_score = - sim_weight.mean(dim=-1).cpu().numpy()
                test_anomaly_scores.append(anomaly_score)
                test_labels.append(y)

        test_anomaly_scores = np.concatenate(test_anomaly_scores)
        test_labels = np.concatenate(test_labels)
        return roc_auc_score(test_labels, test_anomaly_scores)
