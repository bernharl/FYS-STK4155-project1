import numpy as np
import matplotlib.pyplot as plt
from main import OrdinaryLeastSquares


degrees = np.arange(0, 11)
pred_error = np.zeros_like(degrees, dtype=float)
pred_error_train = np.zeros_like(pred_error)

for i in degrees:
    OLS = OrdinaryLeastSquares(degree=i, stddev=0.1,terrain_data=False, filename="SRTM_data_Kolnes_Norway.tif", path="datafiles/",)
    pred_error[i], pred_error_train[i] = OLS.k_fold(k=5, calc_train=True)

plt.plot(degrees, np.log10(pred_error), label="Test", color="r")
plt.plot(degrees, np.log10(pred_error_train), label="Train", color="g")
plt.xlabel("Model Complexity [polynomial degree]")
plt.ylabel("Prediction Error")
#plt.ylim([0, 0.14])
"""plt.text(
    0.02, 0.13, "High bias\nLow variance\n<------", fontsize=10, verticalalignment="top"
)
plt.text(
    8, 0.13, "Low bias\nHigh variance\n------>", fontsize=10, verticalalignment="top"
)"""
plt.legend(loc=3)
plt.savefig("../doc/figs/biasvariancetradeoff.eps")
plt.show()
