import argparse
import json
from downstream.utils import THINGSFeatureTransform
from downstream.anomaly_detection.evaluation import ADEvaluator
from downstream.anomaly_detection.text_evaluation import ADZeroShotEvaluator
from utils.probing.helpers import model_name_to_thingsvision
from tqdm.auto import tqdm


def main(dataset, data_root, source, model_name, module, path_to_transforms,
         module_type, num_classes, output_file, shift_indices, archive_path=None,
         device='cuda', knn_k=5, clip_zero_shot=False):
    output = {
        "dataset": dataset,
        "model": model_name,
        "source": source,
        "module": module,
        "results": []
    }
    name, model_params = model_name_to_thingsvision(model_name)
    options = dict(dataset=dataset, model_name=name, module=module,
                   source=source, device=device,
                   model_params=model_params,
                   data_dir=data_root
                   )
    if dataset == 'cifar100-shift':
        options["train_indices"] = shift_indices

    if clip_zero_shot:
        evaluator = ADZeroShotEvaluator(**options)
    else:
        evaluator = ADEvaluator(**options)
    results = evaluator.evaluate(things_transform=None,
                                 normal_classes=list(range(num_classes)), knn_k=knn_k)
    output["baseline"] = results
    for path_to_transform in tqdm(path_to_transforms):
        things_transform = THINGSFeatureTransform(source=source, model_name=model_name,
                                                  module=module_type,
                                                  archive_path=archive_path,
                                                  path_to_transform=path_to_transform,
                                                  device=device)
        results = evaluator.evaluate(things_transform=things_transform,
                                     normal_classes=list(range(num_classes)),
                                     knn_k=knn_k)
        output["results"].append({
            "path_to_transform": path_to_transform,
            "results": results
        })
        with open(output_file, 'w+') as f:
            json.dump(output, f)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/spaces/datasets")
    parser.add_argument("--dataset", default='cifar10')
    parser.add_argument("--model", default='resnet18')
    parser.add_argument("--module", default='avgpool')
    parser.add_argument('--module-type', default='penultimate')
    parser.add_argument("--source", default='torchvision')
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument('--archive')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--clip-zero-shot', action='store_true')
    parser.add_argument(
        "--transform_paths",
        default=["/home/space/datasets/things/transforms/transforms_without_norm.pkl"],
        nargs='+'
    )
    parser.add_argument('--shift-indices', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument("--out")
    args = parser.parse_args()

    ad_results = main(dataset=args.dataset, data_root=args.data_root,
                      source=args.source,
                      model_name=args.model, module=args.module,
                      module_type=args.module_type,
                      archive_path=args.archive,
                      device=args.device,
                      output_file=args.out,
                      path_to_transforms=args.transform_paths,
                      num_classes=args.classes,
                      shift_indices=args.shift_indices,
                      knn_k=args.k,
                      clip_zero_shot=args.clip_zero_shot)
