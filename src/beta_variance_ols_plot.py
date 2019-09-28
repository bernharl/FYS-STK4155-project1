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

# OLS beta variance for Franke function
ols_franke = OrdinaryLeastSquares(degree=4, stddev=0.1)
ols_franke.regression_method()
x_axis = np.arange(len(ols_franke.beta))
fig, ax = plt.subplots()
fig.set_size_inches(2.942, 1.818)
ax.errorbar(
    x_axis, ols_franke.beta, yerr=np.sqrt(ols_franke.beta_variance), fmt=".", capsize=1
)
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$\beta_n$")
ax.grid()
ax.set_xticks(range(0, len(ols_franke.beta), 4))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
fig.tight_layout()
fig.savefig(
    "../doc/figs/beta_variance_ols_Franke.pdf",
    pad_inches=0,
    bbox_inches="tight",
    dpi=1000,
)

# OLS beta variance for terrain data
ols_terrain = OrdinaryLeastSquares(
    degree=10,
    terrain_data=True,
    filename="SRTM_data_LakeTanganyika_Africa.tif",
    path="datafiles/",
    skip_x_terrain=4,
    skip_y_terrain=4,
)
ols_terrain.regression_method()
x_axis = np.arange(len(ols_terrain.beta))
fig, ax = plt.subplots()
fig.set_size_inches(2.942, 1.818)
ax.errorbar(
    x_axis,
    ols_terrain.beta,
    yerr=np.sqrt(ols_terrain.beta_variance),
    fmt=".",
    capsize=1,
)
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$\beta_n$")
ax.grid()
ax.set_xticks(range(0, len(ols_terrain.beta), 4))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
fig.tight_layout()
fig.savefig(
    "../doc/figs/beta_variance_ols_terrain.pdf",
    pad_inches=0,
    bbox_inches="tight",
    dpi=1000,
)
