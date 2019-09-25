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
    stddev=0.1,
)
ols_franke.regression_method()
x_axis = np.arange(len(ols_franke.beta))
fig, ax = plt.subplots()
fig.set_size_inches(  2.9,  1.81134774961)
ax.errorbar(x_axis, ols_franke.beta, yerr=np.sqrt(ols_franke.beta_variance), fmt=".", capsize=1)
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$\beta_n$")
ax.grid()
ax.set_xticks(range(0, len(ols_franke.beta), 2))
fig.tight_layout()
fig.savefig("../doc/figs/beta_variance_ols_Franke.pdf", dpi=1000)


ols_terrain = OrdinaryLeastSquares(
    degree=5,
    terrain_data=True,
    filename="SRTM_data_LakeTanganyika_Africa.tif",
    path="datafiles/",
    skip_x_terrain=150,
    skip_y_terrain=150,
)
ols_terrain.regression_method()
x_axis = np.arange(len(ols_terrain.beta))
fig, ax = plt.subplots()
fig.set_size_inches(  2.9,  1.81134774961)
ax.errorbar(x_axis, ols_terrain.beta, yerr=np.sqrt(ols_terrain.beta_variance), fmt=".", capsize=1)
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$\beta_n$")
ax.grid()
fig.tight_layout()
ax.set_xticks(range(0, len(ols_terrain.beta), 2))
fig.savefig("../doc/figs/beta_variance_ols_terrain.pdf", dpi=1000)
