from main import OrdinaryLeastSquares, RidgeRegression, LassoRegression

# --------- R2 scores for Franke function stddev=0.1 and terrain data .---------
R2_train = []
R2_test = []

ols_franke = OrdinaryLeastSquares(stddev=0.1, degree=4)
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

ridge_franke = RidgeRegression(stddev=0.1, degree=4)
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


lasso_franke = LassoRegression(stddev=0.1, degree=4, lambd=1e-4)
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

# ---------------------- For noisy data, stddev=1 --------------------
R2_train_n = []
R2_test_n = []

ols_franke_n = OrdinaryLeastSquares(stddev=1,degree=4)
ols_franke_n.regression_method()
R2_train_n.append(ols_franke_n.r_squared_train)
R2_test_n.append(ols_franke_n.r_squared)
del ols_franke_n

ridge_franke_n = RidgeRegression(stddev=1, degree=4, lambd=1e-4)
ridge_franke_n.regression_method()
R2_train_n.append(ridge_franke_n.r_squared_train)
R2_test_n.append(ridge_franke_n.r_squared)
del ridge_franke_n

lasso_franke_n = LassoRegression(stddev=1, degree=4, lambd=1e-4)
lasso_franke_n.regression_method()
R2_train_n.append(lasso_franke_n.r_squared_train)
R2_test_n.append(lasso_franke_n.r_squared)
del lasso_franke_n

reg_type_n = ["OLS", "Ridge", "Lasso"]

print("R2 scores noisy data\n-----------------")
for i in range(len(R2_test_n)):
    print(f"{reg_type_n[i]} training: {R2_train_n[i]:.5f}, {reg_type_n[i]} test: {R2_test_n[i]:.5f}")