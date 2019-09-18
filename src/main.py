from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
import imageio
import scipy as sp
import resource


class RegressionClass:
    def __init__(
        self,
        degree=5,
        stddev=1,
        step=0.05,
        terrain_data=False,
        filename=None,
        path=None,
    ):
        if terrain_data:
            if isinstance(filename, str):
                self.filename = filename
                self.path = path
<<<<<<< HEAD
                self.z_meshgrid = (self.read_image_data())#[::2]
                print(self.z_meshgrid.nbytes / 1024**3)
                #exit()
=======
                self.z_meshgrid = np.asarray(self.read_image_data())#[::1]
                print(type(self.z_meshgrid))
                print(self.z_meshgrid.shape)
                #print(self.z_meshgrid.nbytes / 1024 ** 3)
                # exit()
>>>>>>> 9fcdd54b21d6bca3a9db6a16d6be71292270a34a
                RuntimeWarning(
                    "Given standard deviation is ignored and replaced by the image data's deviations"
                )
                self.stddev = np.std(self.z_meshgrid)
                print(self.z_meshgrid.shape[1])

                x = np.arange(0, self.z_meshgrid.shape[1], dtype="int")
                #print(x.shape)
                #for i in x:
                #    print(i)
                #print(x[0])
                #exit()
                # x = sklpre.scale(x, with_std=False)
                y = np.arange(0, self.z_meshgrid.shape[0], dtype="int")
                #print(len(x), len(y))
                self.x_meshgrid, self.y_meshgrid = np.meshgrid(x, y)
            else:
                raise ValueError("filename must be a string")
        else:
            self.stddev = stddev
            x = np.arange(0, 1, step)
            # x = sklpre.scale(x, with_std=False)
            y = np.arange(0, 1, step)
            # Generate meshgrid data points.
            self.x_meshgrid, self.y_meshgrid = np.meshgrid(x, y)
            self.z_meshgrid = self.noise_function()

        self.x = np.concatenate(self.x_meshgrid)  # .flatten()
        self.y = np.concatenate(self.y_meshgrid)  # .flatten()
        self.z_ = np.concatenate(self.z_meshgrid)  # .flatten()
        # print(np.mean(self.z_))
        self.n = len(self.x)
        self.degree = degree
        self.X = self.design_matrix(self.x, self.y)

        #print(self.z_meshgrid[0,-1])
        print(self.z_meshgrid.shape[0] - 1, x[-1])
        #print(self.X, self.z_, self.z_meshgrid)
        #exit()
        # Split data into training and test set.
        self.X_train, self.X_test, self.z_train, self.z_test = sklms.train_test_split(
            self.X, self.z_, test_size=0.33, shuffle=True
        )
        #print(self.z_train.nbytes / 1024 ** 3, self.X.nbytes / 1024 ** 3)
        # exit()
        # print(np.mean(self.z_train))
        self.modeled = False
        self.terrain_data = terrain_data

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

    def read_image_data(self):
        if self.path == None:
            file_name_path = self.filename
        else:
            file_name_path = self.path + self.filename
        imdata = imageio.imread(file_name_path)
        print(imdata)
        return imdata

    def noise_function(self):
        """
        Adds Gaussian noise with mean zero to the generated data
        """
        f = self.generate_data()
        noise = np.random.normal(0, self.stddev, size=f.shape)
        return f + noise

    def plot_franke(self, plot_data=True):
        """
        3D plot of the Franke function and the linear regression model
        """
        if not plot_data and not self.modeled:
            raise RuntimeError("Please either plot modeled data, real data or both")

        if self.terrain_data:
            skip = 1000
        else:
            skip=1
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        # Plot the surface.
        if self.modeled:
<<<<<<< HEAD
            ax.scatter(self.x[::1000], self.y[::1000], self.regression_model[::1000], s=2, color="black")
            print("Scattered")
        surf = ax.plot_surface(
            self.x_meshgrid[::1000],
            self.y_meshgrid[::1000],
            self.z_meshgrid[::1000],
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
            alpha=0.6,
        )
=======
            ax.scatter(
                self.x[::skip], self.y[::skip], self.regression_model[::skip], s=2, color="black"
            )
            print("Scattered")
        if plot_data:
            surf = ax.plot_surface(
                self.x_meshgrid[::skip],
                self.y_meshgrid[::skip],
                self.z_meshgrid[::skip],
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=False,
                alpha=0.2,
            )
            fig.colorbar(surf, shrink=0.5, aspect=5)
>>>>>>> 9fcdd54b21d6bca3a9db6a16d6be71292270a34a
        print("Surfed")
        # Customize the z axis.
        # ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        # Add a color bar which maps values to colors.

        print("Showing")
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

    def design_matrix(self, x, y):
        """
        Creates the design matrix
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        X = np.zeros((2, len(x)))
        X[0, :] = x
        X[1, :] = y
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
    def eval_model_train(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.X_train @ self.beta

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
        # model_train = self.X_train @ self.beta
        return np.mean((self.z_train - self.eval_model_train) ** 2)

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
        X = self.X_train  # [:, 1:]
        XTX = X.T @ X
        XTz = X.T @ self.z_train
        # Solve XTXbeta = XTz
        # beta = np.zeros_like(self.X_train[0])
        beta = np.linalg.solve(XTX, XTz)
        # beta[0] = np.mean(self.z_train)
        beta_variance = self.stddev ** 2 * np.linalg.inv(XTX)
        self.beta, self.beta_variance_ = beta, np.diag(beta_variance)
        self.modeled = True
        print("USAGE JEDNA")
        #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3)


class RidgeRegression(RegressionClass):
    def __init__(
        self,
        degree=5,
        stddev=1,
        step=0.05,
        lambd=0.1,
        terrain_data=False,
        filename=None,
        path=None,
    ):
        super().__init__(degree, stddev, step, terrain_data, filename, path)
        self.lambd = lambd

    def regression_method(self):
        """
        Uses Ridge regression for given data to calculate regression parameters
        """

        # X = self.X_train[:, 1:] - np.mean(self.X_train[:, 1:], axis=0)

        I = sp.sparse.identity(len(self.X_train[0]) - 1, dtype="int8")
        beta = np.zeros(len(self.X_train[0]))
        # beta[0] = np.mean(self.z_train)
        #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3)
        # exit()

        beta[1:] = np.linalg.solve(
            (self.X_train[:, 1:] - np.mean(self.X_train[:, 1:], axis=0)).T
            @ (self.X_train[:, 1:] - np.mean(self.X_train[:, 1:], axis=0))
            + self.lambd * I,
            (self.X_train[:, 1:] - np.mean(self.X_train[:, 1:], axis=0)).T
            @ (self.z_train - np.mean(self.z_train)),
        )
        # beta[1:] = np.linalg.solve(X.T @ X + self.lambd * I, X.T @ self.z_train)
        self.beta = beta
        self.modeled = True
        #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3)

    @property
    def regression_model(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return ((self.X - np.mean(self.X_train, axis=0)) @ self.beta) + np.mean(
            self.z_train
        )

    @property
    def eval_model(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return ((self.X_test - np.mean(self.X_train, axis=0)) @ self.beta) + np.mean(
            self.z_train
        )

    @property
    def eval_model_train(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return ((self.X_train - np.mean(self.X_train, axis=0)) @ self.beta) + np.mean(
            self.z_train
        )


class LassoRegression(RidgeRegression):
    def regression_method(self):
        """
        Uses LASSO regression for given data to calculate regression parameters
        """
        self.beta = skllm.Lasso(alpha=self.lambd, fit_intercept=False).fit(
            self.X_train[:, 1:] - np.mean(self.X_train[:, 1:], axis=0),
            self.z_train - np.mean(self.z_train, axis=0),
        )
        self.modeled = True

    @property
    def eval_model(self):
        """
        Returns model data using the design matrix and estimated regression parameters
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.beta.predict(self.X_test - np.mean(self.X_train, axis=0)) + np.mean(
            self.z_train
        )

    @property
    def regression_model(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.beta.predict(self.X[:, 1:] - np.mean(self.X_train[:, 1:], axis=0)) + np.mean(
            self.z_train
        )

    @property
    def eval_model_train(self):
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.beta.predict(
            self.X_train - np.mean(self.X_train, axis=0)
        ) + np.mean(self.z_train)
        # return ((self.X_train - np.mean(self.X_train, axis=0)) @ self.beta)  + np.mean(self.z_train)


if __name__ == "__main__":
    # np.random.seed(50)

    ridge = RidgeRegression(
        degree=5,
        stddev=0.1,
        step=0.05,
        lambd=0,
        terrain_data=True,
        filename="SRTM_data_Kolnes_Norway3.tif",
        path="datafiles/",
    )
    ridge.regression_method()
    #ridge.plot_franke()
<<<<<<< HEAD
    
    ols = OrdinaryLeastSquares(
            degree=5,
            stddev=0,
            step=0.05,
            terrain_data=False,
            filename="SRTM_data_Kolnes_Norway3.tif",
            path="datafiles/",
            )
    print(ols.k_fold())
    """

    #print(ridge.beta[0], ols.beta[0])
    #print(np.mean(ols.z_train), np.mean(ridge.z_train))
    #print(ridge.k_fold())
=======
    print(ridge.r_squared)

    """ols = OrdinaryLeastSquares(
        degree=5,
        stddev=0.1,
        step=0.05,
        terrain_data=True,
        filename="SRTM_data_Norway_1.tif",
        path="datafiles/",
    )
    ols.regression_method()
    #ols.plot_franke(False)
    print(ols.r_squared)"""
    #print(ols.regression_model - ridge.regression_model)

    # print(ridge.beta[0], ols.beta[0])
    # print(np.mean(ols.z_train), np.mean(ridge.z_train))
    # print(ridge.k_fold())
>>>>>>> 9fcdd54b21d6bca3a9db6a16d6be71292270a34a
    # test.plot_franke()
    # print(f"MSE {test.mean_squared_error}")
    # print(f"R2 score {test.r_squared}")
    # print(f"Beta variance {test.beta_variance}")
    # test2 = RidgeRegression(degree=5, stddev=0.1, step=0.05, lambd=0)
    # test2.regression_method()
    # test2.plot_franke()

    # test3 = LassoRegression(degree=5, stddev=0.1, step=0.05, lambd=0.001)
    # test3.regression_method()
    # test3.plot_franke()

    """lasso = LassoRegression(
        degree=5,
        stddev=0,
<<<<<<< HEAD
        step=0.05, lambd=5000,
        terrain_data=True,
=======
        step=0.05,
        lambd=1,
        terrain_data=False,
>>>>>>> 9fcdd54b21d6bca3a9db6a16d6be71292270a34a
        filename="SRTM_data_Norway_1.tif",
        path="datafiles/",
    )
    lasso.regression_method()
<<<<<<< HEAD
    lasso.plot_franke()
=======
    lasso.plot_franke()"""
>>>>>>> 9fcdd54b21d6bca3a9db6a16d6be71292270a34a
