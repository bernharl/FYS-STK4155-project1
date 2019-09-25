import numpy as np
import matplotlib.pyplot as plt
from main import OrdinaryLeastSquares, RidgeRegression, LassoRegression


fonts = {
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

plt.rcParams.update(fonts)

# Prediction error for OLS regression
degrees = np.arange(0, 20)

pred_error = np.zeros_like(degrees, dtype=float)
pred_error_train = np.zeros_like(pred_error)

for i in degrees:
    print(i)
    OLS = OrdinaryLeastSquares(
        degree=i,
        stddev=0.1,
        terrain_data=True,
        filename="SRTM_data_LakeTanganyika_Africa.tif",
        path="datafiles/",
    )
    pred_error[i], pred_error_train[i] = OLS.k_fold(k=5, calc_train=True)
    del OLS
pred_log = np.log10(pred_error)
pred_log_train = np.log10(pred_error_train)
fig, ax = plt.subplots()
fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
ax.plot(degrees, pred_log_train, label="Train", color="g")
ax.plot(degrees, pred_log, linestyle="--", label="Test", color="r")
ax.set_xlabel("Model Complexity [polynomial degree]")
ax.set_ylabel(r"log$_{10}$(Prediction Error)")
ax.set_ylim(
    [
        np.min(pred_log_train) - np.min(np.abs(pred_log_train)) * 0.1,
        np.max(pred_log) + np.max(np.abs(pred_log)) * 0.3,
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

ax.legend(loc=3)
fig.tight_layout()
fig.savefig("../doc/figs/biasvariancetradeoff_ols_terrain.pdf", dpi=1000)

# Prediction error for Ridge regression

lambda_Ridge = np.logspace(-10, 10, 21)
pred_error_ridge = np.zeros_like(lambda_Ridge)
pred_error_train_ridge = np.zeros_like(pred_error_ridge)

for j, lamb in enumerate(lambda_Ridge):
    ridge_reg = RidgeRegression(
        lambd=lamb,
        terrain_data=True,
        filename="SRTM_data_LakeTanganyika_Africa.tif",
        path="datafiles/",
    )
    pred_error_ridge[j], pred_error_train_ridge[j] = ridge_reg.k_fold(
        k=5, calc_train=True
    )
    del ridge_reg
pred_log = np.log10(pred_error_ridge)
pred_log_train = np.log10(pred_error_train_ridge)

fig, ax = plt.subplots()
fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
ax.plot(np.log10(lambda_Ridge), pred_log_train, label="Train", color="g")
ax.plot(np.log10(lambda_Ridge), pred_log, linestyle="--", label="Test", color="r")
ax.set_xlabel(r"log$_{10}\lambda$")
ax.set_ylabel(r"log$_{10}$(Prediction Error)")
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

ax.legend(loc=3)
fig.tight_layout()
fig.savefig("../doc/figs/biasvariancetradeoff_Ridge_terrain.pdf", dpi=1000)


# Prediction error for LASSO regression
lambda_lasso = np.logspace(0, 4, 8)
pred_error_lasso = np.zeros_like(lambda_lasso)
pred_error_train_lasso = np.zeros_like(pred_error_lasso)

for j, lamb in enumerate(lambda_lasso):
    print(lamb)
    lasso_reg = LassoRegression(
        lambd=lamb,
        terrain_data=True,
        filename="SRTM_data_LakeTanganyika_Africa.tif",
        path="datafiles/",
    )
    pred_error_lasso[j], pred_error_train_lasso[j] = lasso_reg.k_fold(
        k=5, calc_train=True
    )
    del lasso_reg

pred_log = np.log10(pred_error_lasso)
pred_log_train = np.log10(pred_error_train_lasso)

fig, ax = plt.subplots()
fig.set_size_inches(0.9 * 2 * 2.9, 0.9 * 2 * 1.81134774961)
ax.plot(np.log10(lambda_lasso), pred_log_train, label="Train", color="g")
ax.plot(np.log10(lambda_lasso), pred_log, linestyle="--", label="Test", color="r")
ax.set_xlabel(r"log$_{10}\lambda$")
ax.set_ylabel(r"log$_{10}$(Prediction Error)")
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

ax.legend(loc=3)
fig.tight_layout()
fig.savefig("../doc/figs/biasvariancetradeoff_LASSO_terrain.pdf", dpi=1000)
