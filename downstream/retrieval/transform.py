import os
import pickle
import numpy as np


class ThingsFeatureTransform:

    def __init__(self, transform_path='/home/space/datasets/things/transforms/transforms_without_norm.pkl',
                 things_features_path='/home/space/datasets/things/embeddings/model_features_per_source.pkl'):
        with open(os.path.join(transform_path), "rb") as f:
            self.transforms = pickle.load(f)

        with open(things_features_path, "rb") as f:
            self.things_features = pickle.load(f)

    def transform_features(self, features, source='custom', model_name='clip_ViT-B/16', module='penultimate'):
        transform = self.transforms[source][model_name][module]
        things_features_current_model = self.things_features[source][model_name][module]
        things_mean = things_features_current_model.mean()
        things_std = things_features_current_model.std()
        features = (features - things_mean) / things_std
        if 'weights' in transform:
            features = features @ transform["weights"]
            if "bias" in transform:
                features += transform["bias"]
        else:
            features = features @ transform
        return features
