import numpy as np
import matplotlib.pyplot as plt
from main import OrdinaryLeastSquares, RidgeRegression, LassoRegression


# Prediction error for OLS regression
degrees = np.arange(0, 11)

pred_error = np.zeros_like(degrees, dtype=float)
pred_error_train = np.zeros_like(pred_error)

for i in degrees:
    OLS = OrdinaryLeastSquares(
        degree=i,
        stddev=0.1,
        terrain_data=False,
        filename="SRTM_data_Kolnes_Norway.tif",
        path="datafiles/",
    )
    pred_error[i], pred_error_train[i] = OLS.k_fold(k=5, calc_train=True)

"""
plt.plot(degrees, np.log10(pred_error), label="Test", color="r")
plt.plot(degrees, np.log10(pred_error_train), label="Train", color="g")
plt.xlabel("Model Complexity [polynomial degree]")
plt.ylabel("Prediction Error")
# plt.ylim([0, 0.14])
plt.text(
    0.02, 0.13, "High bias\nLow variance\n<------", fontsize=10, verticalalignment="top"
)
plt.text(
    8, 0.13, "Low bias\nHigh variance\n------>", fontsize=10, verticalalignment="top"
)
plt.legend(loc=3)
plt.savefig("../doc/figs/biasvariancetradeoff.eps")
plt.show()
"""

# Prediction error for Ridge regression
lambda_Ridge = np.linspace(0, 20, 20)
pred_error_ridge = np.zeros_like(lambda_Ridge)
pred_error_train_ridge = np.zeros_like(pred_error_ridge)

for j, lamb in enumerate(lambda_Ridge):
    ridge_reg = RidgeRegression(lambd=lamb)
    pred_error_ridge[j], pred_error_train_ridge[j] = ridge_reg.k_fold(
        k=5, calc_train=True
    )

plt.plot(lambda_Ridge, np.log10(pred_error_ridge), label="Test", color="r")
plt.plot(lambda_Ridge, np.log10(pred_error_train_ridge), label="Train", color="g")
plt.xlabel("Model Complexity [polynomial degree]")
plt.ylabel("Prediction Error")
# plt.ylim([0, 0.14])
plt.text(
    0.02, 0.13, "High bias\nLow variance\n<------", fontsize=10, verticalalignment="top"
)
plt.text(
    8, 0.13, "Low bias\nHigh variance\n------>", fontsize=10, verticalalignment="top"
)
plt.legend(loc=3)
plt.savefig("../doc/figs/biasvariancetradeoff_Ridge.eps")
plt.show()


# Prediction error for LASSO regression
lambda_LASSO = np.linspace(0, 1, 10)
