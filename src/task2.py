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
        X = super().design_matrix()
        self.X_train, self.X_test, self.z_train, self.z_test = sklms.train_test_split(
            X, self.z_, test_size=0.33
        )

    def k_fold(self, k):
        """
        Method for k-fold cross-validation
        """
        data = self.z_
        X = self.design_matrix()
        k_fold_data = np.array_split(data, k)
        k_fold_X = np.array_split(X, k)
        index = np.arange(0, self.n, step = 1, dtype="int")
        index_split = np.array_split(index, k)

        for i in range(k):
            # print(index_split[i])
            train_data = k_fold_data[i]
            test_data = np.delete(k_fold_data, i, axis=0).flatten()






if __name__ == "__main__":
    np.random.seed(100)
    h = ResamplingClass().k_fold(5)
