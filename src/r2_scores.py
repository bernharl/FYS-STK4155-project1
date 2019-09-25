from main import OrdinaryLeastSquares, RidgeRegression, LassoRegression


R2_train = []
R2_test = [] 

ols_franke = OrdinaryLeastSquares()
ols_franke.regression_method()
R2_train.append(ols_franke.r_squared_train)
R2_test.append(ols_franke.r_squared)
del ols_franke

ols_terrain = OrdinaryLeastSquares(
    degree=5,
    terrain_data=True,
    filename="SRTM_data_LakeTanganyika_Africa.tif",
    path="datafiles/"
)
ols_terrain.regression_method() 
R2_train.append(ols_terrain.r_squared_train)
R2_test.append(ols_terrain.r_squared)
del ols_terrain
    
ridge_franke = RidgeRegression() 
ridge_franke.regression_method() 
R2_train.append(ridge_franke.r_squared_train)
R2_test.append(ridge_franke.r_squared)
del ridge_franke

ridge_terrain = RidgeRegression(
    degree=5,
    terrain_data=True,
    filename="SRTM_data_LakeTanganyika_Africa.tif",
    path="datafiles/"
    )
ridge_terrain.regression_method() 
R2_train.append(ridge_terrain.r_squared_train)
R2_test.append(ridge_terrain.r_squared)   
del ridge_terrain



lasso_franke = LassoRegression() 
lasso_franke.regression_method() 
R2_train.append(lasso_franke.r_squared_train)
R2_test.append(lasso_franke.r_squared)
del lasso_franke

lasso_terrain = LassoRegression(
    degree=5,
    terrain_data=True,
    filename="SRTM_data_LakeTanganyika_Africa.tif",
    path="datafiles/"
    )
lasso_terrain.regression_method()
R2_train.append(lasso_terrain.r_squared_train)
R2_test.append(lasso_terrain.r_squared)   
del lasso_terrain    

print(R2_test)
    