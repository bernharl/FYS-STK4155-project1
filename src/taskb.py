from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
from main import RegressionClass


class ResamplingClass(RegressionClass):
    def __init__(self, degree=5, stddev=1, step=0.05):
        super().__init__(degree, stddev, step)
        self.X_train, self.X_test, self.z_train, self.z_test = sklms.train_test_split(
            self.X, self.z_, test_size=0.33
        )

    def k_fold(self, k=5):
        """
        Calculates k-fold cross-validation for our data
        """
        already_modeled = self.modeled
        X_train_old, X_test_old, z_train_old, z_test_old = (
            self.X_train,
            self.X_test,
            self.z_train,
            self.z_test,
        )
        index = np.arange(0, self.n, 1)
        index = np.random.choice(index, replace=False, size=len(index))
        index = np.array_split(index, k)
        for i in range(k):
            test_index = index[i]
            train_index = []
            for j in range(k):
                if j != i:
                    train_index.append(index[j])
            train_index = np.array(train_index).flatten()
            self.X_train, self.X_test, self.z_train, self.z_test = (
                self.X[:, train_index],
                self.X[:, test_index],
                self.z_[train_index],
                self.z_[test_indexs],
            )
            self.regression_method()



        self.X_train, self.X_test, self.z_train, self.z_test = (
            X_train_old,
            X_test_old,
            z_train_old,
            z_test_old,
        )
        if already_modeled:
            self.regression_method()

if __name__ == "__main__":
    np.random.seed(100)
    h = ResamplingClass()
    h.k_fold()
