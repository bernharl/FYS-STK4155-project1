from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sklearn.preprocessing as skpre


class RegressionClass:
    def __init__(self, stddev):
        self.x = np.arange(0, 1, 0.05)
        self.y = np.arange(0, 1, 0.05)
        self.stddev = stddev

    def franke_function(self, x, y):
        """
        Creates the Franke function
        """
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4

    def franke_noise(self, x, y):
        """
        Adds Gaussian noise to the Franke function,  ~ N(0,stddev)
        """
        franke = self.franke_function(x, y)
        noise = stddev * np.random.normal(0, self.stddev, size=franke.shape)
        return franke + noise

    def plot_franke(self):
        """
        3D plot of the Franke function
        """
        fig = plt.figure()
        xx, yy = np.meshgrid(self.x, self.y)
        ax = fig.gca(projection="3d")
        z = self.franke_noise(xx, yy)
        # z = self.franke_function()
        # Plot the surface.
        surf = ax.plot_surface(
            self.x, self.y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def design_matrix(self, degree):
        """
        Creates the design matrix
        """
        X = np.zeros((2, len(self.x)))
        X[0, :] = self.x
        X[1, :] = self.y
        X = X.T
        print(X)
        poly = skpre.PolynomialFeatures(degree)
        return poly.fit_transform(X)

    def ordinary_least_squares(self, degree):
        """
        Calculates ordinary least squares regression and the variance of 
        estimated parameters
        """
        X = self.design_matrix(degree)
        z = self.franke_function(self.x, self.y)
        XTX = np.dot(X.T, X)
        XTz = np.dot(X.T, z)
        beta = np.linalg.solve(XTX, XTz)  # solves XTXbeta = XTz
        beta_variance = self.stdd ** 2 * np.inv(XTX)
        return beta, beta_variance

    def mean_squared_error(self):
        pass

    def r_squared(self):
        pass


if __name__ == "__main__":
    np.random.seed(100)
    test = RegressionClass()
    test.ordinary_least_squares(5)
