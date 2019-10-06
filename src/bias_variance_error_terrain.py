import numpy as np
import matplotlib.pyplot as plt
from main import OrdinaryLeastSquares, RidgeRegression, LassoRegression


# Making fonts fit with LateX document.
fonts = {
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
plt.rcParams.update(fonts)

# Degrees to test OLS with.
degrees = np.arange(0, 15)

# Empty arrays for storing train and test errors.
pred_error = np.zeros_like(degrees, dtype=float)
pred_error_train = np.zeros_like(pred_error)
bias_squared = np.zeros_like(pred_error)
variance = np.zeros_like(pred_error)

for i in degrees:
    print(i)
    OLS = OrdinaryLeastSquares(
        degree=i,
        terrain_data=True,
        filename="SRTM_data_LakeTanganyika_Africa.tif",
        path="datafiles/",
        skip_x_terrain=4,
        skip_y_terrain=4,
    )
    pred_error[i], pred_error_train[i], bias_squared[i], variance[i] = OLS.k_fold(
        k=5, calc_train=True, decompose=True
    )
    # Trying to save memory
    del OLS

fig, ax = plt.subplots()
fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
ax.semilogy(degrees, pred_error_train, label="Train", color="g")
ax.semilogy(degrees, pred_error, linestyle="--", label="Test", color="r")
ax.semilogy(degrees, bias_squared, label=r"Bias$^2$")
ax.semilogy(degrees, variance, label="Variance")
ax.set_xlabel("Model Complexity [polynomial degree]")
ax.set_xticks(degrees[::2])
ax.set_ylabel(r"Error")
ax.set_ylim(
    [
        np.min(pred_error_train) - np.min(np.abs(pred_error_train)) * 0.1,
        np.max(pred_error) + np.max(np.abs(pred_error)) * 0.3,
    ]
)

ax.text(
    0.05,
    0.8,
    "High bias\nLow variance\n<------",
    horizontalalignment="left",
    verticalalignment="baseline",
    transform=ax.transAxes,
)
ax.text(
    0.95,
    0.8,
    "Low bias\nHigh variance\n------>",
    horizontalalignment="right",
    verticalalignment="baseline",
    transform=ax.transAxes,
)

ax.legend(loc=6)
fig.tight_layout()
fig.savefig("../doc/figs/biasvariancetradeoff_ols_terrain.pdf", dpi=1000)


# Prediction error for Ridge regression

lambda_Ridge = np.logspace(-10, 10, 21)
pred_error_ridge = np.zeros_like(lambda_Ridge)
pred_error_train_ridge = np.zeros_like(pred_error_ridge)

for j, lamb in enumerate(lambda_Ridge):
    ridge_reg = RidgeRegression(
        degree=10,
        lambd=lamb,
        terrain_data=True,
        filename="SRTM_data_LakeTanganyika_Africa.tif",
        path="datafiles/",
        skip_x_terrain=4,
        skip_y_terrain=4,
    )
    pred_error_ridge[j], pred_error_train_ridge[j] = ridge_reg.k_fold(
        k=5, calc_train=True
    )
    # Trying to save memory
    del ridge_reg

fig, ax = plt.subplots()
fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
ax.loglog(lambda_Ridge, pred_error_ridge_train, label="Train", color="g")
ax.loglog(lambda_Ridge, pred_error_ridge, linestyle="--", label="Test", color="r")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"Error")
ax.set_ylim(
    [
        np.min(pred_log_train) - np.min(np.abs(pred_log_train)) * 0.1,
        np.max(pred_log) + np.max(np.abs(pred_log)) * 0.3,
    ]
)

ax.text(
    0.05,
    0.8,
    "Low bias\nHigh variance\n<------",
    horizontalalignment="left",
    verticalalignment="baseline",
    transform=ax.transAxes,
)
ax.text(
    0.95,
    0.8,
    "High bias\nLow variance\n------>",
    horizontalalignment="right",
    verticalalignment="baseline",
    transform=ax.transAxes,
)

ax.legend(loc=6)
fig.tight_layout()
fig.savefig("../doc/figs/biasvariancetradeoff_Ridge_terrain.pdf", dpi=1000)

# Prediction error for LASSO regression
lambda_lasso = np.logspace(-4, 4, 10)
pred_error_lasso = np.zeros_like(lambda_lasso)
pred_error_train_lasso = np.zeros_like(pred_error_lasso)

for j, lamb in enumerate(lambda_lasso):
    lasso_reg = LassoRegression(
        degree=10,
        lambd=lamb,
        terrain_data=True,
        filename="SRTM_data_LakeTanganyika_Africa.tif",
        path="datafiles/",
        skip_x_terrain=4,
        skip_y_terrain=4,
    )
    pred_error_lasso[j], pred_error_train_lasso[j] = lasso_reg.k_fold(
        k=5, calc_train=True
    )
    # Trying to save memory
    del lasso_reg


fig, ax = plt.subplots()
fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
ax.loglog(lambda_lasso, pred_error_train_lasso, label="Train", color="g")
ax.loglog(lambda_lasso, pred_error_lasso, linestyle="--", label="Test", color="r")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("Error")
ax.set_ylim(
    [
        np.min(pred_error_train_lasso) - np.min(np.abs(pred_error_train_lasso)) * 0.1,
        np.max(pred_error_lasso) + np.max(np.abs(pred_error_lasso)) * 0.3,
    ]
)

ax.text(
    0.05,
    0.8,
    "Low bias\nHigh variance\n<------",
    horizontalalignment="left",
    verticalalignment="baseline",
    transform=ax.transAxes,
)
ax.text(
    0.95,
    0.8,
    "High bias\nLow variance\n------>",
    horizontalalignment="right",
    verticalalignment="baseline",
    transform=ax.transAxes,
)

ax.legend(loc=6)
fig.tight_layout()
fig.savefig("../doc/figs/biasvariancetradeoff_LASSO_terrain.pdf", dpi=1000)
