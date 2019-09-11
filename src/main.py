from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm


class RegressionClass:
    def __init__(self, degree=5, stddev=1, step=0.05, alt_data=None):
        self.stddev = stddev
        x = np.arange(0, 1, step)
        y = np.arange(0, 1, step)
        # Generate meshgrid data points.
        self.x_meshgrid, self.y_meshgrid = np.meshgrid(x, y)
        self.z_meshgrid = self.noise_function()
        self.x = self.x_meshgrid.flatten()
        self.y = self.y_meshgrid.flatten()
        self.z_ = self.z_meshgrid.flatten()
        self.n = len(self.x)
        self.degree = degree
        self.X = self.design_matrix()
        # Split data into training and test set.
        self.X_train, self.X_test, self.z_train, self.z_test = sklms.train_test_split(
            self.X, self.z_, test_size=0.33
        )
        self.modeled = False
        if alt_data == None:
            pass
        elif callable(alt_data):
            self.generate_data = alt_data
        else:
            raise ValueError("alt_data must be either None or callable")


    def generate_data(self):
        """
        Generates data using the Franke function
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
        Adds Gaussian noise with mean zero to the generated data
        """
        f = self.generate_data()
        noise = np.random.normal(0, self.stddev, size=f.shape)
        return f + noise

    def plot_franke(self):
        """
        3D plot of the Franke function and the linear regression model
        """
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        # Plot the surface.
        if self.modeled:
            ax.scatter(self.x, self.y, self.regression_model, s=2, color="black")
        surf = ax.plot_surface(
            self.x_meshgrid,
            self.y_meshgrid,
            self.z_meshgrid,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
            alpha=0.5
        )
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def k_fold(self, k=5, calc_train=False):
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
        index = np.arange(0, len(X_train_old[:, 0]), 1)
        index = np.random.choice(index, replace=False, size=len(index))
        index = np.array_split(index, k)
        mse = np.zeros(k)
        if calc_train:
            mse_train = np.zeros_like(mse)

        for i in range(k):
            test_index = index[i]
            train_index = []
            for j in range(k):
                if j != i:
                    train_index.append(index[j])
            train_index = np.concatenate(train_index)
            self.X_train, self.X_test, self.z_train, self.z_test = (
                X_train_old[train_index, :],
                X_train_old[test_index, :],
                z_train_old[train_index],
                z_train_old[test_index],
            )
            self.regression_method()
            mse[i] = self.mean_squared_error
            if calc_train:
                mse_train[i] = self.mean_squared_error_train

        mse = np.mean(mse)
        if calc_train:
            mse_train = np.mean(mse_train)

        self.X_train, self.X_test, self.z_train, self.z_test = (
            X_train_old,
            X_test_old,
            z_train_old,
            z_test_old,
        )
        if already_modeled:
            self.regression_method()

        if calc_train:
            return mse, mse_train

        return mse

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
    def regression_model(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.X @ self.beta

    @property
    def eval_model(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.X_test @ self.beta

    @property
    def mean_squared_error(self):
        """
        Calculates the MSE for chosen regression model
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return np.mean((self.z_test - self.eval_model) ** 2)

    @property
    def mean_squared_error_train(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        model_train = self.X_train @ self.beta
        return np.mean((self.z_train - model_train) ** 2)

    @property
    def r_squared(self):
        """
        Calculates R2 score for chosen regression model
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        z = self.z_test
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
        X = self.X_train
        XTX = X.T @ X
        XTz = X.T @ self.z_train
        # Solve XTXbeta = XTz
        beta = np.linalg.solve(XTX, XTz)
        beta_variance = self.stddev ** 2 * np.linalg.inv(XTX)
        self.beta, self.beta_variance_ = beta, np.diag(beta_variance)
        self.modeled = True


class RidgeRegression(RegressionClass):
    def __init__(self, degree=5, stddev=1, step=0.05, lambd=0.1, alt_data=None):
        super().__init__(degree, stddev, step, alt_data)
        self.lambd = lambd

    def regression_method(self):
        """
        Uses Ridge regression for given data to calculate regression parameters
        """
        X = self.X_train[:, 1:] - np.mean(self.X_train[:, 1:], axis=0)
        I = np.identity(len(self.X_train[1]) - 1)
        beta = np.zeros(len(self.X_train[1]))
        beta[0] = np.mean(self.z_train)  # appears to be wrong, need further inspection
        beta[1:] = np.linalg.solve(X.T @ X + self.lambd * I, X.T @ self.z_train)
        self.beta = beta
        self.modeled = True


class LassoRegression(RidgeRegression):
    def regression_method(self):
        """
        Uses LASSO regression for given data to calculate regression parameters
        """
        self.beta = skllm.Lasso(alpha=self.lambd).fit(self.X_train, self.z_train)
        self.modeled = True

    @property
    def eval_model(self):
        """
        Returns model data using the design matrix and estimated regression parameters
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.beta.predict(self.X_test)

    @property
    def regression_model(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.beta.predict(self.X)


if __name__ == "__main__":
    np.random.seed(50)
    # test = OrdinaryLeastSquares(degree=5, stddev=0.1, step=0.05)
    # test.regression_method()
    # test.plot_franke()
    # test.k_fold(10)
    # test.plot_franke()
    # print(f"MSE {test.mean_squared_error}")
    # print(f"R2 score {test.r_squared}")
    # print(f"Beta variance {test.beta_variance}")
    test2 = RidgeRegression(degree=5, stddev=0.1, step=0.05, lambd=0)
    test2.regression_method()
    test2.plot_franke()

    # test3 = LassoRegression(degree=5, stddev=0.1, step=0.05, lambd=0.001)
    # test3.regression_method()
    # test3.plot_franke()
