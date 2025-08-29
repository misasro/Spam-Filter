import numpy as np


class PCA:
    """
    Reduce dimension using principle component analysis
    """
    def __init__(self, n_components):
        """
        :param n_components: number of columns of transformed dataset
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X) -> 'PCA':
        """
        Calculate centre of dataset, make SVD decomposition, find principle components
        :param X: dataset
        :return: self
        """
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n_components].T
        return self

    def transform(self, X):
        """
        Transforms the dataset
        :param X: dataset
        :return: matrix with `n_components` columns
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
