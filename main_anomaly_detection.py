import argparse
import json
import torch
import torchvision
from downstream.utils import THINGSFeatureTransform
from downstream.anomaly_detection.evaluation import ADEvaluator


def get_model(name, module):
    assert name.startswith('resnet'), 'currently this implementation only supports resnet models'
    model = getattr(torchvision.models, name)(pretrained=True)
    if module == 'penultimate':
        model.fc = torch.nn.Identity()
    model.eval()
    return model


def main(dataset, data_root, model_name, module, path_to_transform, do_transform=False, device='cuda'):
    model = get_model(name=model_name, module=module)

    things_transform = None
    if do_transform:
        things_transform = THINGSFeatureTransform(source='torchvision', model_name=model_name, module=module,
                                                  path_to_transform=path_to_transform)
    results = []
    options = {}
    if dataset == 'cifar100-shift':
        options = dict(train_indices=[0, 1, 2])
    for cls in range(0, 20):
        evaluator = ADEvaluator(dataset=dataset, model=model,
                                normal_cls=cls, device=device,
                                things_transform=things_transform,
                                data_dir=data_root, **options)
        auc = evaluator.evaluate(do_transform=do_transform)
        results.append(auc)
    return {
        "dataset": dataset,
        "model": model_name,
        "module": module,
        "path_to_transform": path_to_transform,
        "do_transform": do_transform,
        "results": results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/spaces/datasets")
    parser.add_argument("--dataset", default='cifar10')
    parser.add_argument("--model", default='resnet18')
    parser.add_argument("--module", default='penultimate')
    parser.add_argument(
        "--transform_path",
        default="/home/space/datasets/things/transforms/transforms_without_norm.pkl",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    ad_results = main(dataset=args.dataset, data_root=args.data_root,
                      model_name=args.model, module=args.module,
                      path_to_transform=args.transform_path)
    with open(args.out, 'w+') as f:
        json.dump(ad_results, f)
