import matplotlib.pyplot as plt
import numpy as np
from main import OrdinaryLeastSquares, RidgeRegression, LassoRegression



# OLS model plots
ols_franke = OrdinaryLeastSquares(stddev=0.1, degree=4)
ols_franke.regression_method()
ols_franke.plot_model("OLS")
# Trying to save memory
del ols_franke

ols_terrain = OrdinaryLeastSquares(
    skip_x_terrain=4,
    skip_y_terrain=4,
    degree=10,
    terrain_data=True,
    filename="SRTM_data_LakeTanganyika_Africa.tif",
    path="datafiles/",
)
ols_terrain.regression_method()
ols_terrain.plot_model("OLS")
# Trying to save memory
del ols_terrain

# Ridge model plots
ridge_franke = RidgeRegression(stddev=0.1, degree=4, lambd=1e-5)
ridge_franke.regression_method()
ridge_franke.plot_model("Ridge")
# Trying to save memory
del ridge_franke

ridge_terrain = RidgeRegression(
    skip_x_terrain=4,
    skip_y_terrain=4,
    stddev=0.1,
    degree=10,
    lambd=1e-10,
    terrain_data=True,
    filename="SRTM_data_LakeTanganyika_Africa.tif",
    path="datafiles/",
)
ridge_terrain.regression_method()
ridge_terrain.plot_model("Ridge")
# Trying to save memory
del ridge_terrain


# Lasso model plots
lasso_franke = LassoRegression(stddev=0.1, degree=4, lambd=1e-5)
lasso_franke.regression_method()
lasso_franke.plot_model("Lasso")
del lasso_franke

lasso_terrain = LassoRegression(
    skip_x_terrain=4,
    skip_y_terrain=4,
    stddev=0.1,
    degree=10,
    lambd=1e-4,
    terrain_data=True,
    filename="SRTM_data_LakeTanganyika_Africa.tif",
    path="datafiles/",
)
lasso_terrain.regression_method()
lasso_terrain.plot_model("Lasso")
# Trying to save memory
del lasso_terrain
