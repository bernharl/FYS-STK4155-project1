from task2 import ResamplingClass
from main import RegressionClass
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms


class PlaceholderName(ResamplingClass):
    def __init__(self, degree=5, stddev=1, step=0.05, lambd=1):
        super().__init__(degree, stddev, step)
        self.lambd = lambd

    def ridge_regression(self):
        """
        Just some pseudo-ish code for now before class just to set up the method. 
        I haven't checked any of the dimensions, so please remember to take
        an extra look at those 
        """
        I = np.identity("fill")
        self.beta_rigde = np.linalg.solve(
            np.dot(X.T, X) + self.lambd * I, np.dot(X.T, z)
        )


if __name__ == "__main__":
    c = PlaceholderName().ridge_regression()
