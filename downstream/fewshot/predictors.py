from datetime import datetime
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsRegressor

from downstream.fewshot.tip_adapter import TipAdapter

Array = np.ndarray


def train_regression(
    train_targets: Array, train_features: Array, k: int = None, solver: str = "lbfgs"
):
    n_train = train_features.shape[0]
    print("N. train:", n_train)
    start_t = datetime.now()

    reg = LogisticRegressionCV(
        Cs=(1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6),
        fit_intercept=True,
        penalty="l2",
        cv=k,
        max_iter=500,
        solver=solver,
    )

    reg.fit(train_features, train_targets)

    print("Finished training. Elapsed time:", datetime.now() - start_t)

    return reg


def train_knn(train_targets: Array, train_features: Array, k: int = 1):
    n_train = train_features.shape[0]
    print("N. train:", n_train)
    start_t = datetime.now()

    reg = KNeighborsRegressor(
        n_neighbors=k,
        algorithm="ball_tree",
    )

    reg.fit(train_features, train_targets)

    print("Finished training. Elapsed time:", datetime.now() - start_t)

    return reg


def train_tip(train_targets: Array, train_features: Array, zero_shot_weights: Array):
    start_t = datetime.now()

    one_hot_targets = np.zeros((train_targets.size, train_targets.max() + 1))
    one_hot_targets[np.arange(train_targets.size), train_targets] = 1
    F = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
    W = zero_shot_weights / np.linalg.norm(zero_shot_weights, axis=1, keepdims=True)

    # The default values appear to work best
    alpha_list = [1.]
    beta_list = [5.5]
    best_reg = None
    best_beta, best_alpha, best_acc = 0, 0, 0

    for beta in beta_list:
        for alpha in alpha_list:
            reg = TipAdapter(F=F, W=W, L=one_hot_targets, alpha=alpha, beta=beta)
            acc, _ = test_regression(reg, train_targets, train_features)
            if acc > best_acc:
                best_acc = acc
                best_beta = beta
                best_alpha = alpha
                best_reg = reg

    print("Finished training. Elapsed time:", datetime.now() - start_t)
    print("\tBest beta: %.3f, best alpha: %.3f, best acc: %.3f" % (best_beta, best_alpha, best_acc))

    return best_reg


def test_regression(
    regressor,
    test_targets: Array,
    test_features: Array,
):
    n_test = test_features.shape[0]
    print("N. test:", n_test)

    preds = regressor.predict(test_features)
    acc = np.sum([p == t for p, t in zip(preds, test_targets)]) / len(preds)
    try:
        regularization_strength = regressor.C_
        print("Accuracy: %.3f, Regularization:" % acc, regularization_strength)
    except AttributeError:
        print("Accuracy: %.3f" % acc)

    return acc, preds


def get_regressor(
    train_features: Array,
    train_targets: Array,
    regressor_type: str,
    k: Optional[int] = None,
    solver: str = "lbfgs",
    zero_shot_weights: Optional[Array] = None
):
    if regressor_type == "ridge":
        regressor = train_regression(train_targets, train_features, k=k, solver=solver)
    elif regressor_type == "knn":
        regressor = train_knn(train_targets, train_features)
    elif regressor_type == "tip":
        regressor = train_tip(train_targets, train_features, zero_shot_weights=zero_shot_weights)
    else:
        raise ValueError(f"Unknown regressor: {regressor_type}")
    return regressor
