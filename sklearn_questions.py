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
number of samples corectly classified). You need to implement the `fit`,
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
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

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
from sklearn.utils.multiclass import unique_labels


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
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.X_ = X
        self.y_ = y

        self.classes_ = unique_labels(self.y_)
        self.n_features_in_ = self.X_.shape[1]

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
        # checking if the model has been trained before
        X = check_array(X)
        # input validation
        idx_eucl = pairwise_distances(X, self.X_)
        # compute euclidean distance
        idx_eucl = np.argsort(idx_eucl)[:, 0:self.n_neighbors]
        # find the indices of the k closest samples
        class_predictions = self.y_[idx_eucl]
        # extract the class labels of the closest samples
        predicted_classes = []
        for y_pred_single in class_predictions:
            unique_classes, indices = np.unique(
                y_pred_single, return_inverse=True)
            predicted_classes.append(
                unique_classes[np.argmax(np.bincount(indices))])
            # find the most common label
        return np.array(predicted_classes)

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
        check_is_fitted(self)
        # Checking if the model has been trained before
        X, y = check_X_y(X, y)
        # Checking if X and y are under the good format
        predicted_classes = self.predict(X)
        # Predicting the classes
        count_correct = 0
        # Initializing the count of the number of correct predictions
        for i in range(predicted_classes.shape[0]):
            if predicted_classes[i] == y[i]:
                count_correct += 1
        # The score metric correspond to the number of correct predictions
        # compared to the false predictions
        return count_correct/predicted_classes.shape[0]


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
        # check if the data is pandas series and convert it to dataframe
        if type(X) == pd.Series:
            X = X.to_frame()
        # check if the index is not range index and reset the index
        if type(X.index) != pd.RangeIndex:
            X = X.reset_index()
        # check if the time_col is of datetime format
        if not np.issubdtype(X[self.time_col].dtype, np.datetime64):
            raise ValueError('datetime')
        # return the number of splits by re-sampling on a monthly basis
        return len(X.resample('M', on=self.time_col)) - 1

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
        # check if the data is pandas series and convert it to dataframe
        if type(X) == pd.Series:
            X = X.to_frame()
        # reset the index of the dataframe
        X = X.reset_index()
        # get the number of splits
        n_splits = self.get_n_splits(X, y, groups)
        # re-sample the dataframe on a monthly basis
        X_resampled = X.resample('M', on=self.time_col)
        # create a helper function to get the indexes of the
        # resampled dataframe

        def index_array(array):
            return array.index
        # get the indexes of the resampled dataframe
        idx_month = X_resampled.apply(index_array)
        # iterate through all the splits
        for i in range(n_splits):
            idx_train = idx_month.iloc[i]
            idx_test = idx_month.iloc[i+1]
            # assign the indexes of the current month as the training set and
            # the next month as the test set
            yield (
                idx_train.values, idx_test.values
            )
