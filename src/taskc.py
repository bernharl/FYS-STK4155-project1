from taskb import ResamplingClass
from main import RegressionClass
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as skllm



class PlaceholderName(ResamplingClass):
    def __init__(self, degree=5, stddev=1, step=0.05, lambd=0.1):
        super().__init__(degree, stddev, step)
        self.lambd = lambd

    def ridge_regression(self):
        """
        Calculates Ridge regression 
        """
        X = self.X[:,1:]
        I = np.identity(len(self.X[1])-1)
        beta = np.zeros(len(self.X[1]))
        beta[0] = np.mean(self.z_)
        beta[1:] = np.linalg.solve(
            np.dot(X.T, X) + self.lambd * I, np.dot(X.T, self.z_)
        )
        self.beta = beta
        self.modeled = True

    def lasso_regression(self):
        """
        Calculates LASSO regression
        """
        self.beta = skllm.Lasso(alpha=self.lambd).fit(self.X, self.z_)
        self.modeled = True
        


if __name__ == "__main__":
    np.random.seed(100)
    c = PlaceholderName()
    c.ridge_regression()
    print(c.eval_model - c.z_)
