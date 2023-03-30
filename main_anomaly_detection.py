import argparse
import json
from downstream.utils import THINGSFeatureTransform
from downstream.anomaly_detection.evaluation import ADEvaluator
from utils.probing.helpers import model_name_to_thingsvision


def main(dataset, data_root, source, model_name, module, path_to_transform,
         module_type,
         num_classes, archive_path=None,
         do_transform=False, device='cuda'):
    things_transform = None
    if do_transform:
        things_transform = THINGSFeatureTransform(source=source, model_name=model_name, module=module_type,
                                                  archive_path=archive_path,
                                                  path_to_transform=path_to_transform)
    options = {}
    if dataset == 'cifar100-shift':
        options = dict(train_indices=[0, 1, 2])

    name, model_params = model_name_to_thingsvision(model_name)
    evaluator = ADEvaluator(dataset=dataset, model_name=name, module=module,
                            source=source, device=device,
                            things_transform=things_transform,
                            model_params=model_params,
                            data_dir=data_root, **options)

    results = evaluator.evaluate(do_transform=do_transform, normal_classes=list(range(num_classes)))
    return {
        "dataset": dataset,
        "model": model_name,
        "source": source,
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
    parser.add_argument("--module", default='avgpool')
    parser.add_argument('--module-type', default='penultimate')
    parser.add_argument("--source", default='torchvision')
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--transform", action="store_true")
    parser.add_argument('--archive')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        "--transform_path",
        default="/home/space/datasets/things/transforms/transforms_without_norm.pkl",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    ad_results = main(dataset=args.dataset, data_root=args.data_root,
                      source=args.source,
                      model_name=args.model, module=args.module,
                      module_type=args.module_type,
                      do_transform=args.transform,
                      archive_path=args.archive,
                      device=args.device,
                      path_to_transform=args.transform_path, num_classes=args.classes)
    with open(args.out, 'w+') as f:
        json.dump(ad_results, f)
