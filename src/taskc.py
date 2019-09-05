from taskb import ResamplingClass
from main import RegressionClass
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm


"""
I haven't checked the dimensions very throroughly, so please remember 
to take an extra look at those 
"""

class PlaceholderName(ResamplingClass):
    def __init__(self, degree=5, stddev=1, step=0.05, lambd=0.1):
        super().__init__(degree, stddev, step)
        self.lambd = lambd


    def ridge_regression(self):

        I = np.identity(len(self.X[1]))
        self.beta = np.linalg.solve(
            np.dot(self.X.T, self.X) + self.lambd * I, np.dot(self.X.T, self.z_)
        )
        self.modeled = True


    def lasso_regression(self):
        self.beta = skllm.Lasso(alpha=self.lambd).fit(self.X, self.z_)
        self.modeled = True
     

if __name__ == "__main__":
    c = PlaceholderName().ridge_regression()
    d = PlaceholderName().lasso_regression()
