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
        data = self.z_






if __name__ == "__main__":
    np.random.seed(100)
    h = ResamplingClass()
    h.k_fold()
