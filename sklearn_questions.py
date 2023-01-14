import numpy as np
from collections import Counter
from sklearn.model_selection import BaseCrossValidator

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
# @@ -58,6 +58,7 @@
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(BaseEstimator, ClassifierMixin):

    """
    KNearestNeighbors classifier."""
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
        self.X_, self.y_ = check_X_y(X, y)
        check_classification_targets(self.y_)
        self.classes_ = unique_labels(self.y_)
        self.n_features_in_ = self.X_.shape[1]
        return self

    def predict(self, X):

        """
        Predict function.
        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.
        Returns
        ----------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """

        y_pred = np.zeros(X.shape[0])
        return y_pred
        check_is_fitted(self)
        X = check_array(X)
        y_pred = []
        distances = pairwise_distances(self.X_, X)
        for i in range(distances.shape[1]):
            yOrdered = self.y_[distances[:, i].argsort()]
            yClose = yOrdered[:self.n_neighbors]
            countClose = Counter(yClose)
            y_pred.append(countClose.most_common(1)[0][0])

        return np.array(y_pred)

    def score(self, X, y):

        """
        Calculate the score of the prediction.
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

        return 0.
        check_classification_targets(y)
        yPred = self.predict(X)
        return (yPred == y).sum()/len(y)


class MonthlySplit(BaseCrossValidator):

    """
    CrossValidator based on monthly split.
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

        """
        Return the number of splitting iterations in the cross-validator.
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

        return 0
        X = X.reset_index()
        if not (np.dtype('datetime64[ns]') == X[self.time_col].dtype):
            raise ValueError("datetime")
        allYM = np.unique(X[self.time_col].dt.to_period('M'))
        return len(allYM) - 1

    def split(self, X, y, groups=None):

        """
        Generate indices to split data into training and test set.
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

        n_samples = X.shape[0]
        X = X.reset_index()
        n_splits = self.get_n_splits(X, y, groups)
        X["periodYMDates"] = X[self.time_col].dt.to_period('M')
        allYM = np.sort(np.unique(X[self.time_col].dt.to_period('M')))
        for i in range(n_splits):
            idx_train = range(n_samples)
            idx_test = range(n_samples)
            X_split_train = X.loc[(X["periodYMDates"] == allYM[i])]
            idx_train = X_split_train.index.values
            X_split_test = X.loc[(X["periodYMDates"] == allYM[i+1])]
            idx_test = X_split_test.index.values
            yield (
                idx_train, idx_test
            )
