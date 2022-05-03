# Copyright (c) 2022. Bommarito Consulting, LLC
# SPDX-License-Identifier: Apache-2.0

# standard library imports
from typing import Union, Callable, List, Optional

# package imports
import numpy
import catboost

# local imports
from rfcorr.sign import sign_without_zero


def get_corr_classification(
    X: numpy.array,
    y: numpy.array,
    num_iterations: Optional[int] = None,
    num_trees: Optional[int] = None,
    learning_rate: Optional[float] = None,
    max_depth: Optional[int] = None,
    cat_features: Union[List[int], numpy.array] = None,
    task_type: str = "CPU",
    random_state: numpy.random.RandomState = None,
):
    """
    Get a random forest correlation with classifier-based feature importance
    based on features X on regression target y.

    :param X: The input array.
    :param y: The output array.
    :param num_iterations: number of iterations
    :param num_trees: The number of decision trees in the random forest ensemble.
    :param learning_rate: The catboost learning rate
    :param max_depth: The maximum depth of the tree.
    :param cat_features: categorical feature indices
    :param task_type: whether to use GPU or CPU
    :param random_state: numpy random state to use.
    :return: The correlation between X and y.

    """
    cb_model = catboost.CatBoostClassifier(
        iterations=num_iterations,
        n_estimators=num_trees,
        max_depth=max_depth,
        learning_rate=learning_rate,
        cat_features=cat_features,
        task_type=task_type,
        random_state=random_state,
    ).fit(X, y, cat_features=cat_features, verbose=False)

    # handle permutation importance case
    return cb_model.feature_importances_


def get_corr_regression(
    X: numpy.array,
    y: numpy.array,
    loss_function: str = "RMSE",
    num_iterations: Optional[int] = None,
    num_trees: Optional[int] = None,
    max_depth: Optional[int] = None,
    learning_rate: Optional[float] = None,
    cat_features: Union[List[int], numpy.array] = None,
    task_type: str = "CPU",
    random_state: numpy.random.RandomState = None,
):
    """
    Get a random forest correlation with classifier-based feature importance
    based on features X on regression target y.

    :param X: The input array.
    :param y: The output array.
    :param loss_function: which loss function to use; see https://catboost.ai/en/docs/concepts/loss-functions-regression
    :param num_iterations: number of iterations
    :param num_trees: The number of decision trees in the random forest ensemble.
    :param max_depth: The maximum depth of the tree.
    :param learning_rate: The catboost learning rate
    :param cat_features: categorical feature indices
    :param task_type: whether to use GPU or CPU
    :param random_state: numpy random state to use.
    :return: The correlation between X and y.

    """
    cb_model = catboost.CatBoostRegressor(
        loss_function=loss_function,
        iterations=num_iterations,
        n_estimators=num_trees,
        max_depth=max_depth,
        learning_rate=learning_rate,
        cat_features=cat_features,
        task_type=task_type,
        random_state=random_state,
    ).fit(X, y, cat_features=cat_features, verbose=False)

    # handle permutation importance case
    return cb_model.feature_importances_


def get_corr(
    X: numpy.array,
    y: numpy.array,
    method: str = "classification",
    loss_function: Optional[str] = "RMSE",
    num_iterations: Optional[int] = None,
    num_trees: Optional[int] = None,
    max_depth: Optional[int] = None,
    learning_rate: Optional[float] = None,
    cat_features: Union[List[int], numpy.array] = None,
    task_type: str = "CPU",
    random_state: numpy.random.RandomState = None,
):
    """
    Get the "correlation" between array of features X and target column y.

    :param X: The input array.
    :param y: The output array.
    :param method: The method to use: classification or regression
    :param loss_function: if method=regression, which loss function to use
    :param num_iterations: number of iterations
    :param num_trees: The number of decision trees in the random forest ensemble.
    :param max_depth: The maximum depth of the tree.
    :param learning_rate: The catboost learning rate
    :param cat_features: categorical feature indices
    :param task_type: whether to use GPU or CPU
    :param random_state: numpy random state to use.
    :return: The pairwise correlation between X and y.
    """

    # check shape of arrays
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    # check method
    if method not in ["classification", "regression"]:
        raise ValueError("method must be either classification or regression.")
    elif method == "classification":
        return get_corr_classification(
            X=X,
            y=y,
            num_iterations=num_iterations,
            num_trees=num_trees,
            learning_rate=learning_rate,
            max_depth=max_depth,
            cat_features=cat_features,
            task_type=task_type,
            random_state=random_state,
        )
    elif method == "regression":
        return get_corr_regression(
            X=X,
            y=y,
            loss_function=loss_function,
            num_iterations=num_iterations,
            num_trees=num_trees,
            learning_rate=learning_rate,
            max_depth=max_depth,
            cat_features=cat_features,
            task_type=task_type,
            random_state=random_state,
        )


def get_pairwise_corr(
    X: numpy.array,
    method: str = "classification",
    lag: int = 0,
    loss_function: Optional[str] = "RMSE",
    num_iterations: Optional[int] = None,
    num_trees: Optional[int] = None,
    max_depth: Optional[int] = None,
    learning_rate: Optional[float] = None,
    cat_features: Union[List[int], numpy.array] = None,
    task_type: str = "CPU",
    random_state: numpy.random.RandomState = None,
):
    """
    Get the pairwise correlation between all columns in X.

    :param X: The input array.
    :param method: classification or regression
    :param lag: lag to offset features from target
    :param loss_function: if method=regression, which loss function to use
    :param num_iterations: number of iterations
    :param num_trees: The number of decision trees in the random forest ensemble.
    :param max_depth: The maximum depth of the tree.
    :param learning_rate: The catboost learning rate
    :param cat_features: categorical feature indices
    :param task_type: whether to use GPU or CPU
    :param random_state: numpy random state to use.
    :return: The pairwise correlation between X and y.
    """
    # check method
    if method not in ["classification", "regression", "auto"]:
        raise ValueError("method must be either auto, classification, or regression.")

    # check that we have sufficient samples for the lag
    if lag > X.shape[0] + 2:
        raise ValueError("X must have at least `lag + 2` rows")
    if lag < 0:
        raise ValueError("lag must be >= 0")

    # initialize return matrix
    corr_mat = numpy.eye(X.shape[1])

    # get row indices
    lag_target_index = list(range(lag, X.shape[0]))
    lag_feature_index = list(range(0, X.shape[0] - lag))

    # iterate through each target column
    for target_index in range(X.shape[1]):
        # setup feature column index
        feature_index = [i for i in range(X.shape[1]) if i != target_index]
        if cat_features is not None:
            cat_feature_index = [
                (i < target_index) * i + (i > target_index) * (i - 1)
                for i in cat_features
                if i != target_index
            ]
        else:
            cat_feature_index = None

        # determine method to use if variable types are mixed
        if method == "auto":
            if cat_features is not None:
                target_method = (
                    "classification" if target_index in cat_features else "regression"
                )
            else:
                if X[:, target_index].dtype in [
                    numpy.float64,
                    numpy.float32,
                    numpy.int64,
                    numpy.int32,
                ]:
                    target_method = "regression"
                else:
                    target_method = "classification"
        else:
            target_method = method

        # set into matrix
        XX = X[lag_feature_index, :][:, feature_index]
        yy = X[lag_target_index, :][:, target_index]

        # check constant-valued yy
        if numpy.unique(yy).shape[0] > 1:
            r = get_corr(
                X=XX,
                y=yy,
                method=target_method,
                loss_function=loss_function,
                num_iterations=num_iterations,
                num_trees=num_trees,
                max_depth=max_depth,
                learning_rate=learning_rate,
                cat_features=cat_feature_index,
                task_type=task_type,
                random_state=random_state,
            )
            corr_mat[target_index, feature_index] = r
        else:
            corr_mat[target_index, feature_index] = numpy.nan

    return corr_mat
