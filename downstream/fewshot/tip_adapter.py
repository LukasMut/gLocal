import numpy as np

Array = np.ndarray


class TipAdapter:
    def __init__(
        self, F: Array, L: Array, W: Array, alpha: float = 1, beta: float = 5.5
    ):
        self.F = F
        self.L = L
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)
        self.alpha = alpha
        self.beta = beta

    def predict(self, test_features):
        test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)

        clip_logits = test_features @ self.W.T

        affinity = test_features @ self.F.T
        cache_logits = np.exp(-self.beta * (1 - affinity)) @ self.L

        tip_logits = clip_logits + cache_logits * self.alpha
        preds = np.argmax(tip_logits, axis=1)

        return preds
