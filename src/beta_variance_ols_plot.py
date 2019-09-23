import matplotlib.pyplot as plt
import numpy as np
from main import OrdinaryLeastSquares

fonts = {
        "font.family": "serif",
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
}

plt.rcParams.update(fonts)

ols_franke = OrdinaryLeastSquares(
    degree=5,
    stddev=0.1
)
ols_franke.regression_method()
x_axis = np.arange(len(ols_franke.beta))
fig, ax = plt.subplots()
fig.set_size_inches(2 * 2.9, 2 * 1.81134774961)
ax.errorbar(x_axis, ols_franke.beta, yerr=np.sqrt(ols_franke.beta_variance), fmt="o")
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$\beta_n$")
ax.grid()
fig.savefig("../doc/figs/beta_variance_ols_Franke.eps", dpi=1000)


ols_terrain = OrdinaryLeastSquares(
    degree=2,
    terrain_data=True,
    filename="SRTM_data_Norway_2.tif",
    path="datafiles/"
)
ols_terrain.regression_method()
ols_terrain.plot_model()
print(ols_terrain.r_squared)
exit()
x_axis = np.arange(len(ols_terrain.beta))
fig, ax = plt.subplots()
fig.set_size_inches(2 * 2.9, 2 * 1.81134774961)
ax.errorbar(x_axis, ols_terrain.beta, yerr=np.sqrt(ols_terrain.beta_variance), fmt="o")
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$\beta_n$")
ax.grid()
fig.savefig("../doc/figs/beta_variance_ols_terrain.eps", dpi=1000)
