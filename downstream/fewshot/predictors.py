from datetime import datetime
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsRegressor

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


def regress(
    train_targets: Array,
    train_features: Array,
    test_targets: Array,
    test_features: Array,
    k: int = None,
    regressor: str = "ridge",
    solver: str = "lbfgs",
):
    if regressor == "ridge":
        reg = train_regression(train_targets, train_features, k, solver=solver)
    elif regressor == "knn":
        reg = train_knn(train_targets, train_features)
    else:
        raise ValueError(f"Unknown regressor: {regressor}")
    acc, preds = test_regression(reg, test_targets, test_features)
    return acc, preds


def get_regressor(
    train_features: Array,
    train_targets: Array,
    regressor_type: str,
    k: Optional[int] = None,
    solver: str = "lbfgs",
):
    if regressor_type == "ridge":
        regressor = train_regression(train_targets, train_features, k=k, solver=solver)
    elif regressor_type == "knn":
        regressor = train_knn(train_targets, train_features)
    else:
        raise ValueError(f"Unknown regressor: {regressor_type}")
    return regressor
