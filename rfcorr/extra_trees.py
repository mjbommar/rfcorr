# Copyright (c) 2022. Bommarito Consulting, LLC
# SPDX-License-Identifier: Apache-2.0

# standard library imports
from typing import Union, Callable

# package imports
import numpy
import sklearn.ensemble

# local imports
from .sign import sign_without_zero


def get_corr_classification(X: numpy.array,
                            y: numpy.array,
                            transform: Callable = sign_without_zero,
                            num_trees: int = 100,
                            criterion: str = "gini",
                            max_features: Union[str, int, float] = "auto",
                            max_depth: int = None,
                            bootstrap: bool = True,
                            random_state: numpy.random.RandomState = None):
    """
    Get an extra trees forest correlation with classifier-based feature importance
    based on features X on regression target y.

    :param X: The input array.
    :param y: The output array.
    :param transform: A function to transform the label array if it is continuous
    :param num_trees: The number of decision trees in the random forest ensemble.
    :param criterion: The function to measure the quality of a split.
    :param max_features: The number of features for the random forest model.
    :param max_depth: The maximum depth of the tree.
    :param bootstrap: Whether to use bootstrap sampling.
    :param random_state: numpy random state to use.
    :return: The correlation between X and y.

    """
    et_model = sklearn.ensemble.ExtraTreesClassifier(n_estimators=num_trees,
                                                     criterion=criterion,
                                                       max_features=max_features,
                                                       max_depth=max_depth,
                                                       bootstrap=bootstrap,
                                                       random_state=random_state) \
        .fit(X, transform(y))
    return et_model.feature_importances_


def get_corr_regression(X: numpy.array,
                        y: numpy.array,
                        num_trees: int = 100,
                        criterion: str = "squared_error",
                        max_features: Union[str, int, float] = "auto",
                        max_depth: int = None,
                        bootstrap: bool = True,
                        random_state: numpy.random.RandomState = None):
    """
    Get an extra trees forest correlation with classifier-based feature importance
    based on features X on classification labels y.

    :param X: The input array.
    :param y: The output array.
    :param num_trees: The number of decision trees in the random forest ensemble.
    :param criterion: The function to measure the quality of a split.
    :param max_features: The number of features for the random forest model.
    :param max_depth: The maximum depth of the tree.
    :param random_state: numpy random state to use.
    :param bootstrap: Whether to use bootstrap sampling.
    :return: The correlation between X and y.
    """
    et_model = sklearn.ensemble.ExtraTreesRegressor(n_estimators=num_trees,
                                                    criterion=criterion,
                                                      max_features=max_features,
                                                      max_depth=max_depth,
                                                      bootstrap=bootstrap,
                                                      random_state=random_state) \
        .fit(X, y)
    return et_model.feature_importances_


def get_corr(X: numpy.array,
             y: numpy.array,
             method: str = "classification",
             num_trees: int = 100,
             criterion: str = "gini",
             max_features: Union[str, int, float] = "auto",
             max_depth: int = None,
             bootstrap: bool = True,
             random_state: numpy.random.RandomState = None):
    """
    Get the "correlation" between array of features X and target column y.

    :param X: The input array.
    :param y: The output array.
    :param method: The method to use: classification or regression
    :param num_trees: The number of decision trees in the random forest ensemble.
    :param criterion: The function to measure the quality of a split.
    :param max_features: The number of features for the random forest model.
    :param max_depth: The maximum depth of the tree.
    :param bootstrap: Whether to use bootstrap sampling.
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
        return get_corr_classification(X=X,
                                       y=y,
                                       transform=sign_without_zero,
                                       num_trees=num_trees,
                                       criterion=criterion,
                                       max_features=max_features,
                                       max_depth=max_depth,
                                       bootstrap=bootstrap,
                                       random_state=random_state)
    elif method == "regression":
        return get_corr_regression(X=X,
                                   y=y,
                                   num_trees=num_trees,
                                   criterion=criterion,
                                   max_features=max_features,
                                   max_depth=max_depth,
                                   bootstrap=bootstrap,
                                   random_state=random_state)


def get_pairwise_corr(X: numpy.array,
                      method: str = "classification",
                      lag: int = 0,
                      num_trees: int = 100,
                      criterion: str = "gini",
                      max_features: Union[str, int, float] = "auto",
                      max_depth: int = None,
                      bootstrap: bool = True,
                      random_state: numpy.random.RandomState = None):
    """
    Get the pairwise correlation between all columns in X.

    :param X: The input array.
    :param method: The method to use: classification or regression
    :param num_trees: The number of decision trees in the random forest ensemble.
    :param criterion: The function to measure the quality of a split.
    :param max_features: The number of features for the random forest model.
    :param max_depth: The maximum depth of the tree.
    :param bootstrap: Whether to use bootstrap sampling.
    :param random_state: numpy random state to use.
    :return: The pairwise correlation between X and y.
    """
    # check method
    if method not in ["classification", "regression"]:
        raise ValueError("method must be either classification or regression.")

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

        # set into matrix
        XX = X[lag_feature_index, :][:, feature_index]
        yy = X[lag_target_index, :][:, target_index]
        corr_mat[target_index, feature_index] = get_corr(X=XX,
                                                         y=yy,
                                                         method=method,
                                                         num_trees=num_trees,
                                                         criterion=criterion,
                                                         max_features=max_features,
                                                         max_depth=max_depth,
                                                         bootstrap=bootstrap,
                                                         random_state=random_state)

    return corr_mat
