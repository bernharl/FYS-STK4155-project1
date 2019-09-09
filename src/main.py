from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm

class RegressionClass:
    def __init__(self, degree=5, stddev=1, step=0.05):
        x = np.arange(0, 1, step)
        y = np.arange(0, 1, step)
        self.stddev = stddev
        self.x_meshgrid, self.y_meshgrid = np.meshgrid(x, y)
        self.z_meshgrid = self.noise_function()
        self.x = self.x_meshgrid.flatten()
        self.y = self.y_meshgrid.flatten()
        self.z_ = self.z_meshgrid.flatten()
        self.n = len(self.x)
        self.degree = degree
        self.X = self.design_matrix()
        self.modeled = False


    def generate_data(self):
        """
        Creates the Franke function
        """
        x = self.x_meshgrid
        y = self.y_meshgrid
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4

    def noise_function(self):
        """
        Adds Gaussian noise to the function f,  ~ N(0,stddev)
        """
        f = self.generate_data()
        noise = np.random.normal(0, self.stddev, size=f.shape)
        return f + noise

    def plot_franke(self):
        """
        3D plot of the Franke function
        """
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        # Plot the surface.
        if self.modeled:
            ax.scatter(self.x, self.y, self.eval_model)
        surf = ax.plot_surface(
            self.x_meshgrid, self.y_meshgrid, self.z_meshgrid, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def design_matrix(self):
        """
        Creates the design matrix
        """
        X = np.zeros((2, self.n))
        X[0, :] = self.x
        X[1, :] = self.y
        X = X.T
        poly = sklpre.PolynomialFeatures(self.degree)
        return poly.fit_transform(X)

    def regression_method(self):
        raise RuntimeError("Please do not use this class directly!")

    @property
    def eval_model(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.X @ self.beta

    @property
    def mean_squared_error(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return np.sum((self.z_ - self.eval_model) ** 2) / self.n

    @property
    def r_squared(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        z = self.z_
        return 1 - np.sum((z - self.eval_model) ** 2) / np.sum((z - np.mean(z)) ** 2)

    @property
    def beta_variance(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.beta_variance_

class OrdinaryLeastSquares(RegressionClass):
    def regression_method(self):
        """
        Calculates ordinary least squares regression and the variance of
        estimated parameters
        """
        X = self.X
        XTX = X.T @ X
        XTz = X.T @ self.z_
        beta = np.linalg.solve(XTX, XTz)  # solves XTXbeta = XTz
        beta_variance = self.stddev ** 2 * np.linalg.inv(XTX)
        self.beta, self.beta_variance_ = beta, np.diag(beta_variance)
        self.modeled = True

class RidgeRegression(RegressionClass):
    def __init__(self, degree=5, stddev=1, step=0.05, lambd=0.1):
        super().__init__(degree, stddev, step)
        self.lambd = lambd

    def regression_method(self):
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


class LassoRegression(RidgeRegression):
    def regression_method(self):
        """
        Calculates LASSO regression
        """
        self.beta = skllm.Lasso(alpha=self.lambd).fit(self.X, self.z_)
        self.modeled = True

    @property
    def eval_model(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.beta.predict(self.X)



if __name__ == "__main__":
    np.random.seed(100)
    test = OrdinaryLeastSquares(degree=5, stddev=0.1, step=0.05)
    test.regression_method()
    test.plot_franke()
    print(f"MSE {test.mean_squared_error}")
    print(f"R2 score {test.r_squared}")
    print(f"Beta variance {test.beta_variance}")
