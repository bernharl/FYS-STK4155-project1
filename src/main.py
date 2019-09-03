from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed


class RegressionClass:
    def __init__(self):
        x = np.arange(0, 1, 0.05)
        y = np.arange(0, 1, 0.05)
        self.x, self.y = np.meshgrid(x, y)

    def franke_function(self):

        term1 = 0.75 * np.exp(-(0.25 * (9 * self.x - 2) ** 2) - 0.25 * ((9 * self.y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * self.x + 1) ** 2) / 49.0 - 0.1 * (9 * self.y + 1))
        term3 = 0.5 * np.exp(-(9 * self.x - 7) ** 2 / 4.0 - 0.25 * ((9 * self.y - 3) ** 2))
        term4 = -0.2 * np.exp(-(9 * self.x - 4) ** 2 - (9 * self.y - 7) ** 2)
        return term1 + term2 + term3 + term4

    def franke_noise(self, stddev):
        franke = self.franke_function()
        noise = stddev * np.random.normal(1, stddev, size=franke.shape)
        return franke + noise


    def plot_franke(self):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        z = self.franke_noise(0.5)
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
test = RegressionClass()
test.plot_franke()
