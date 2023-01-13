"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples correctly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `check_*` functions imported in the file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datetime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
November 2020 to March 2021, you have have 4 splits. The first split
will allow to learn on November data and predict on December data, the
second split to learn December and predict on January etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors
        super().__init__()

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
            The current instance of the classifier.
        """
        # Check that X and y have correct shape & dtypes
        X, y = check_X_y(X, y)

        # Check the training target
        check_classification_targets(y)

        # Fit the classifier by filling fitted attributes
        self.classes_ = np.unique(y)
        self._fit_y = y
        self._fit_X = X
        self.n_features_in_ = X.shape[1]

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
        # Checks
        # Check if fit has been called
        check_is_fitted(self)

        # Check the input
        X = check_array(X)

        # Predict classes
        # D[i, j] = euclidean distance from X_train[i] to X_predict[j]
        # shape: (n_train_samples, n_test_samples)
        D = pairwise_distances(self._fit_X, X)

        # Fill predictions vector
        y_pred = np.empty(shape=X.shape[0], dtype=self._fit_y.dtype)

        for j in range(X.shape[0]):
            n_smallest_dists_idx = np.argpartition(D[:, j], self.n_neighbors)
            n_closest_classes = self._fit_y[
                n_smallest_dists_idx[: self.n_neighbors]
            ]
            classes, counts = np.unique(n_closest_classes, return_counts=True)
            predicted_class = classes[np.argmax(counts)]
            y_pred[j] = predicted_class

        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            Target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        accuracy = (self.predict(X) == y).sum() / y.shape[0]
        return accuracy


class MonthlySplit(BaseCrossValidator):
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

    def __init__(self, time_col="index"):  # noqa: D107
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
        # Retrieve time data & check
        if self.time_col == "index":
            time_col_data = X.index
        else:
            time_col_data = X[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(time_col_data):
            raise ValueError("time_col should be of datetime type!")

        # Compute the number of splits
        maxi = time_col_data.max()
        mini = time_col_data.min()
        # start by counting the number of months
        # from whole calendar years
        nb_whole_years = maxi.year - mini.year - 1
        if nb_whole_years > 0:
            nb_months_from_whole_years = nb_whole_years * 12
        else:
            nb_months_from_whole_years = 0
        # then add the leading or trailing months
        # (count the starting and ending months of the data as
        # a whole month as suggested by the detailed instructions)
        if maxi.year == mini.year:
            nb_leading_months = maxi.month - mini.month + 1
            nb_trailing_months = 0
        else:
            nb_leading_months = 12 - mini.month + 1
            nb_trailing_months = maxi.month
        # total nb of months
        total_nb_month = (
            nb_leading_months +
            nb_months_from_whole_years +
            nb_trailing_months
        )
        # get n_splits
        n_splits = total_nb_month - 1
        return n_splits

    def _date_adder_helper(self, month, year):
        """Add a month to the given (month, year) pair.

        Parameters
        ----------
        month : int
            The month number.
        year :int
            The year number.

        Returns
        -------
        month : int
            The new month number.
        year :int
            The new year number.
        """
        if month == 12:
            new_date = (1, year + 1)
        else:
            new_date = (month + 1, year)
        return new_date

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
        # Retrieve time data
        if self.time_col == "index":
            time_col_data = X.index
        else:
            time_col_data = pd.DatetimeIndex(X[self.time_col])

        # Initialize
        # n_samples = X.shape[0]
        n_splits = self.get_n_splits(X, y, groups)
        train_month, train_year = (
            time_col_data.min().month,
            time_col_data.min().year,
        )
        train_mask = (time_col_data.year == train_year) & (
            time_col_data.month == train_month
        )
        idx_train = X.reset_index().loc[train_mask].index.to_numpy()

        # Loop through splits
        for _ in range(n_splits):
            test_month, test_year = self._date_adder_helper(
                train_month,
                train_year
            )
            test_mask = (time_col_data.year == test_year) & (
                time_col_data.month == test_month
            )
            idx_test = X.reset_index().loc[test_mask].index.to_numpy()
            # yield indices
            yield (idx_train, idx_test)
            # update
            train_month, train_year = test_month, test_year
            idx_train = idx_test.copy()
