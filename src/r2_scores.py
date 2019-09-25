from main import OrdinaryLeastSquares, RidgeRegression, LassoRegression


R2_train = []
R2_test = []

ols_franke = OrdinaryLeastSquares(stddev=0.1)
ols_franke.regression_method()
R2_train.append(ols_franke.r_squared_train)
R2_test.append(ols_franke.r_squared)
del ols_franke

ols_terrain = OrdinaryLeastSquares(
    skip_x_terrain=4,
    skip_y_terrain=4,
    degree=5,
    terrain_data=True,
    filename="SRTM_data_LakeTanganyika_Africa.tif",
    path="datafiles/",
)
ols_terrain.regression_method()
R2_train.append(ols_terrain.r_squared_train)
R2_test.append(ols_terrain.r_squared)
del ols_terrain

ridge_franke = RidgeRegression(stddev=0.1)
ridge_franke.regression_method()
R2_train.append(ridge_franke.r_squared_train)
R2_test.append(ridge_franke.r_squared)
del ridge_franke

ridge_terrain = RidgeRegression(
    skip_x_terrain=4,
    skip_y_terrain=4, 
    stddev=0.1, 
    degree=5,
    terrain_data=True,
    filename="SRTM_data_LakeTanganyika_Africa.tif",
    path="datafiles/",
)
ridge_terrain.regression_method()
R2_train.append(ridge_terrain.r_squared_train)
R2_test.append(ridge_terrain.r_squared)
del ridge_terrain


lasso_franke = LassoRegression(stddev=0.1, lambd=1e-4)
lasso_franke.regression_method()
R2_train.append(lasso_franke.r_squared_train)
R2_test.append(lasso_franke.r_squared)
del lasso_franke

lasso_terrain = LassoRegression(
    skip_x_terrain=4,
    skip_y_terrain=4,
    stddev=0.1,
    degree=5,
    terrain_data=True,
    filename="SRTM_data_LakeTanganyika_Africa.tif",
    path="datafiles/",
)
lasso_terrain.regression_method()
R2_train.append(lasso_terrain.r_squared_train)
R2_test.append(lasso_terrain.r_squared)
del lasso_terrain


reg_type = [
    "OLS Franke",
    "OLS Terrain",
    "Ridge Franke",
    "Ridge Terrain",
    "Lasso Franke",
    "Lasso Terrain",
]

for i in range(len(R2_test)):
    print(reg_type[i])
    print(f"R2 training: {R2_train[i]:.3f}, R2 test: {R2_test[i]:.3f}")
