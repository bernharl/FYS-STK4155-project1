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
            X, self.z_, train_size=0.66
        )

    def k_fold(self, k):
        """
        Method for k-fold cross-validation
        """
        pass
        



if __name__ == "__main__":
    np.random.seed(100)
    h = ResamplingClass()
