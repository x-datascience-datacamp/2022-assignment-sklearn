"""sklearn questions."""

from abc import ABC

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import (
    check_classification_targets, unique_labels)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

         Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        ----------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        self.X_train_, self.y_train_ = check_X_y(X, y)
        check_classification_targets(self.y_train_)
        self.classes_ = unique_labels(self.y_train_)
        self.n_features_in_ = self.X_train_.shape[1]
        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        ----------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self)
        X = check_array(X)
        pair_dist = pairwise_distances(X, self.X_train_)
        ind_k_neigh = np.argpartition(pair_dist,
                                      kth=self.n_neighbors,
                                      axis=-1)[:, :self.n_neighbors]
        label_k_neigh = self.y_train_[ind_k_neigh]
        axis = 1
        u, indices = np.unique(label_k_neigh, return_inverse=True)
        y_pred = u[
            np.argmax(
                np.apply_along_axis(
                    np.bincount,
                    axis,
                    indices.reshape(label_k_neigh.shape),
                    None,
                    np.max(indices) + 1
                ),
                axis=axis)
        ]
        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)


class MonthlySplit(BaseCrossValidator, ABC):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        X = X.reset_index()
        date = X[self.time_col]
        if not isinstance(X[self.time_col][0], pd.Timestamp):
            raise ValueError("We don't have a datetime column")
        return date.dt.to_period('M').nunique() - 1

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        n_splits = self.get_n_splits(X, y, groups)
        X = X.reset_index()
        months = X[self.time_col].dt.to_period('M')
        months_sorted = np.sort(months.unique())
        for i in range(n_splits):
            idx_train = X[months == months_sorted[i]].index.to_list()
            idx_test = X[months == months_sorted[i + 1]].index.to_list()
            yield (
                idx_train, idx_test
            )
