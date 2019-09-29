from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, ScalarFormatter
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
import imageio
import scipy as sp
import resource
import matplotlib


class RegressionClass:
    def __init__(
        self,
        degree=5,
        stddev=1,
        n_points=20,
        terrain_data=False,
        filename=None,
        path=None,
        skip_x_terrain=1,
        skip_y_terrain=1,
    ):
        """
        Base class for all regression methods. Please do not call on its own, as it
        does not contain any regression methods by itself.

        Parameters:

        input(int, default 5): What polynomial degree to run linear regression with.

        stddev(float, default 1): What standard deviation to apply to the synthetic
                    noise added to Franke's function. Ignored if terrain_data==True.

        n_points(int, default 20): How many x- and y-points used to generate Franke's
                                   function data. Total data points will be n^2.

        terrain_data(bool, default False): If True, then custom terrain data is used
                                           instead of Franke's function.

        filename(str, default None): Filename of terrain data,
                                     only used if terrain_data==True.

        path(str, default None): Path of terrain data, only used if terrain_data==True.

        skip_x_terrain(int, default 1): Defines how many points to skip in terrain
                                        data in the x-direction, only used if
                                        terrain_data==True.

        skip_y_terrain(int, default 1): Defines how many points to skip in terrain
                                        data in the y-direction, only used if
                                        terrain_data==True.
        """
        if terrain_data:
            if isinstance(filename, str):
                self.filename = filename
                self.path = path
                # Reading data, skiping points in x and y direction
                self.z_meshgrid = np.asarray(self.read_image_data(), dtype=np.int16)[
                    ::skip_y_terrain, ::skip_x_terrain
                ]
                RuntimeWarning(
                    "Given standard deviation is ignored and replaced by the image data's standar deviation"
                )
                self.stddev = np.std(self.z_meshgrid)

                # Defining x and y
                x = np.arange(0, self.z_meshgrid.shape[1]) / (
                    self.z_meshgrid.shape[1] - 1
                )
                y = np.arange(0, self.z_meshgrid.shape[0]) / (
                    self.z_meshgrid.shape[0] - 1
                )
                # Meshgridding, needed for plot and to get all combinations
                self.x_meshgrid, self.y_meshgrid = np.meshgrid(x, y)
            else:
                raise ValueError("filename must be a string")
        else:
            self.stddev = stddev
            # Defining x and y
            x = np.linspace(0, 1, n_points, endpoint=True)
            y = np.linspace(0, 1, n_points, endpoint=True)

            # Generate meshgrid data points.
            self.x_meshgrid, self.y_meshgrid = np.meshgrid(x, y)
            self.z_meshgrid = self.noise_function()

        # Flattening inputs and data to make it possible to create design matrix
        self.x = np.concatenate(self.x_meshgrid)
        self.y = np.concatenate(self.y_meshgrid)
        self.z_ = np.concatenate(self.z_meshgrid)
        self.degree = degree
        # Creating design matrix
        self.X = self.design_matrix(self.x, self.y)
        # Split data into training and test set.
        self.X_train, self.X_test, self.z_train, self.z_test = sklms.train_test_split(
            self.X, self.z_, test_size=0.33, shuffle=True
        )
        self.modeled = False
        self.terrain_data = terrain_data # This is a boolean

    def generate_data(self):
        """
        Generates data using Franke's function.
        """
        x = self.x_meshgrid
        y = self.y_meshgrid
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4

    def read_image_data(self):
        """
        Reads terrain data from file. Only used if self.terrain_data==True.
        """
        if self.path == None:
            file_name_path = self.filename
        else:
            file_name_path = self.path + self.filename
        imdata = imageio.imread(file_name_path)
        return imdata

    def noise_function(self):
        """
        Adds Gaussian noise with mean zero to
         the data generated with Franke's function.
        """
        f = self.generate_data()
        # Gaussian noise
        noise = np.random.normal(0, self.stddev, size=f.shape)
        return f + noise

    def plot_model(self, method="", plot_data=True):
        """
        3D plot of the regression model, also (optional) plots the data used.

        Parameters:

        method(str, default ""): Part of filename for saved plots.

        plot_data(bool, default True): Whether to plot the data used.
        """
        if not plot_data and not self.modeled:
            raise RuntimeError("Please either plot modeled data, real data or both")
        # Cannot plot all data if using terrain data, would crash matplotlib
        if self.terrain_data:
            skip = 1000
        else:
            skip = 1
        fig = plt.figure()
        # Figure size set to equal the width of the report, using golden ratio for height.
        fig.set_size_inches(2 * 2.942, 2 * 1.818)
        fig.tight_layout()
        ax = fig.gca(projection="3d")
        # If modeled, plot scatterplot of model
        if self.modeled:
            ax.scatter(
                self.x[::skip],
                self.y[::skip],
                self.regression_model[::skip],
                s=2,
                color="black",
            )
        # Plot the data modeled.
        if plot_data:
            surf = ax.plot_surface(
                self.x_meshgrid,
                self.y_meshgrid,
                self.z_meshgrid,
                cmap=cm.coolwarm,
                linewidth=0,
                antialiased=False,
                alpha=0.4,
            )
            fig.colorbar(surf, shrink=0.5, aspect=5)
        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(10))

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # Setting nice viewing angle for terrain data
        if self.terrain_data:
            ax.zaxis.set_major_formatter(FormatStrFormatter("%g"))
            zticks = np.linspace(np.min(self.z_), np.max(self.z_), 4)
            ax.set_zticks(zticks)
            ax.view_init(elev=40, azim=105)
            fig.savefig(
                f"../doc/figs/3dmodel_{method}_terrain.pdf",
                pad_inches=0.2,
                bbox_inches="tight",
                dpi=1000,
            )
        # Setting nice viewing angle for Franke data
        else:
            ax.zaxis.set_major_formatter(FormatStrFormatter("%.2g"))

            zticks = np.linspace(np.min(self.z_), np.max(self.z_), 4)
            ax.set_zticks(zticks)
            ax.view_init(elev=20, azim=70)  # good values: 20, 50
            fig.savefig(
                f"../doc/figs/3dmodel_{method}_Franke.pdf",
                pad_inches=0.1,
                bbox_inches="tight",
                dpi=1000,
            )

        plt.close()

    def k_fold(self, k=5, calc_train=False):
        """
        Performs k-fold cross validation for the defined training data.


        Parameters:

        k(int, default 5): What k to use in k-fold cross validation.

        calc_train(bool, default False): Whether to returrn the average training
                                         error in addition to the EPE.


        Returns:

        mse(float): The average test MSE of the folds. Also known as the
                    Expected Prediction Error, EPE.

        mse_train(float): The average training error of the folds, only returned
                          if calc_train==True.
        """
        # This method is implemented in a pretty "hacky" manner, so for it to not
        # change the model if it is already modeled, we have to save the old
        # arrays before overwriting them. This is terribly ineficcient memory-wise,
        # and should not be repeated in the next project.
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
        Creates the design matrix using Scikit-Learn's PolynomialFeatures.

        returns(numpy.ndarray): Design matrix of inputs.
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
        """
        Empty method to make sure the user does not use this class directly.
        """
        raise RuntimeError("Please do not use this class directly!")

    @property
    def regression_model(self):
        """
        Returns the entire regression model. Is for instance used in plotting.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.X @ self.beta

    @property
    def eval_model(self):
        """
        Returns the model applied to the test inputs.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.X_test @ self.beta

    @property
    def eval_model_train(self):
        """
        Returns the model applied to the training inputs.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.X_train @ self.beta

    @property
    def mean_squared_error(self):
        """
        Returns the MSE for the regression model using test data.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return np.mean((self.z_test - self.eval_model) ** 2)

    @property
    def mean_squared_error_train(self):
        """
        Returns the training error of the model.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        # model_train = self.X_train @ self.beta
        return np.mean((self.z_train - self.eval_model_train) ** 2)

    @property
    def r_squared(self):
        """
        Returns the R2 score for the regression model using the test data.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        z = self.z_test
        return 1 - np.sum((z - self.eval_model) ** 2) / np.sum((z - np.mean(z)) ** 2)

    @property
    def r_squared_train(self):
        """
        Returns the R2 score for the regression model using the train data.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        z = self.z_train
        return 1 - np.sum((z - self.eval_model_train) ** 2) / np.sum(
            (z - np.mean(z)) ** 2
        )


class OrdinaryLeastSquares(RegressionClass):
    def regression_method(self):
        """
        Performs the OLS method using inputs and data defined in the class instance.
        Also calculates the variance of the parameters beta.
        """
        X = self.X_train
        XTX = X.T @ X
        XTz = X.T @ self.z_train
        # Solve XTXbeta = XTz
        # Using np.linalg.solve instead of inverting matrix directly, should
        # be more stable.
        beta = np.linalg.solve(XTX, XTz)
        beta_variance = self.stddev ** 2 * np.linalg.inv(XTX)
        self.beta, self.beta_variance_ = beta, np.diag(beta_variance)
        self.modeled = True

    @property
    def beta_variance(self):
        """
        Returns the variance of beta for the OLS regression.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        return self.beta_variance_


class RidgeRegression(RegressionClass):
    def __init__(
        self,
        degree=5,
        stddev=1,
        n_points=20,
        lambd=0.1,
        terrain_data=False,
        filename=None,
        path=None,
        skip_x_terrain=1,
        skip_y_terrain=1,
    ):
        """
        Class for running Ridge Regression. Inherits from RegressionClass.

        Parameters:

        lambda(float, default 0.1): The hyperparameter Lambda.

        For all other parameters, see RegressionClass.
        """
        super().__init__(
            degree,
            stddev,
            n_points,
            terrain_data,
            filename,
            path,
            skip_x_terrain,
            skip_y_terrain,
        )
        self.lambd = lambd

    def regression_method(self):
        """
        Centers inputs and data, then performs Ridge regression.
        """
        I = sp.sparse.identity(len(self.X_train[0]) - 1, dtype="int8")
        beta = np.zeros(len(self.X_train[0]))

        # Using np.linalg.solve instead of inverting matrix directly, should
        # be more stable. Using centered inputs.
        beta[1:] = np.linalg.solve(
            (self.X_train[:, 1:] - np.mean(self.X_train[:, 1:], axis=0)).T
            @ (self.X_train[:, 1:] - np.mean(self.X_train[:, 1:], axis=0))
            + self.lambd * I,
            (self.X_train[:, 1:] - np.mean(self.X_train[:, 1:], axis=0)).T
            @ (self.z_train - np.mean(self.z_train)),
        )
        self.beta = beta
        self.modeled = True

    @property
    def regression_model(self):
        """
        Returns the entire regression model. Is for instance used in plotting.
        This has to be defined separately for Ridge because Ridge needs centered
        inputs.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        # As inputs are centered, we must add the intercept manually.
        return ((self.X - np.mean(self.X_train, axis=0)) @ self.beta) + np.mean(
            self.z_train
        )

    @property
    def eval_model(self):
        """
        Returns the model applied to the test inputs.
        This has to be defined separately for Ridge because Ridge needs centered
        inputs.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        # As inputs are centered, we must add the intercept manually.
        return ((self.X_test - np.mean(self.X_train, axis=0)) @ self.beta) + np.mean(
            self.z_train
        )

    @property
    def eval_model_train(self):
        """
        Returns the model applied to the training inputs.
        This has to be defined separately for Ridge because Ridge needs centered
        inputs.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        # As inputs are centered, we must add the intercept manually.
        return ((self.X_train - np.mean(self.X_train, axis=0)) @ self.beta) + np.mean(
            self.z_train
        )


class LassoRegression(RidgeRegression):
    def regression_method(self):
        """
        Centers inputs and data, then performs LASSO regression using Scikit-Learn.
        """
        self.beta = skllm.Lasso(
            alpha=self.lambd, fit_intercept=False, max_iter=20000, selection="random"
        ).fit(
            self.X_train[:, 1:] - np.mean(self.X_train[:, 1:], axis=0),
            self.z_train - np.mean(self.z_train, axis=0),
        )
        self.modeled = True

    @property
    def eval_model(self):
        """
        Returns the model applied to the test inputs.
        This has to be defined separately for LASSO because uses Scikit-Learn's
        method.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        # As inputs are centered, we must add the intercept manually.
        return self.beta.predict(
            self.X_test[:, 1:] - np.mean(self.X_train[:, 1:], axis=0)
        ) + np.mean(self.z_train)

    @property
    def regression_model(self):
        """
        Returns the entire regression model. Is for instance used in plotting.
        This has to be defined separately for LASSO because uses Scikit-Learn's
        method.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        # As inputs are centered, we must add the intercept manually.
        return self.beta.predict(
            self.X[:, 1:] - np.mean(self.X_train[:, 1:], axis=0)
        ) + np.mean(self.z_train)

    @property
    def eval_model_train(self):
        """
        Returns the model applied to the training inputs.
        This has to be defined separately for LASSO because uses Scikit-Learn's
        method.
        """
        if not self.modeled:
            raise RuntimeError("Run a regression method first!")
        # As inputs are centered, we must add the intercept manually.      
        return self.beta.predict(
            self.X_train[:, 1:] - np.mean(self.X_train[:, 1:], axis=0)
        ) + np.mean(self.z_train)
