import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
import sklearn.metrics as skm
from main import OrdinaryLeastSquares, RidgeRegression, LassoRegression


# Test of our OLS and Ridge regression methods from main.py, as well as the
# k-fold cross validation method. We have also executed tests on our methods 
# for the R2 score and the MSE. 
# All tests use artificial data from Franke's function from main.py

# -------------------- test of OLS regression -----------------------------
def test_OLS():
    """
    Testing that our OLS implementation works the same as Scikit-Learn's within
    a relative tolerance 1e-5.
    """
    OLS = OrdinaryLeastSquares(stddev=0, n_points=50)
    # scikit-learn method:
    OLS_scikit = skllm.LinearRegression().fit(OLS.X_train, OLS.z_train)
    evalmodel_scikit = OLS_scikit.predict(OLS.X_test)
    # our method:
    OLS.regression_method()
    evalmodel_test = OLS.eval_model

    np.testing.assert_allclose(evalmodel_scikit, evalmodel_test, rtol=1e-5)


# -------------------- test of Ridge regression --------------------------
def test_ridge():
    """
    Testing that our Ridge implementation works the same as Scikit-Learn's within
    a relative tolerance 1e-5.
    """
    ridge = RidgeRegression(stddev=0, n_points=50)
    # scikit-learn method:
    ridge_scikit = skllm.Ridge(alpha=ridge.lambd).fit(ridge.X_train, ridge.z_train)
    evalmodel_scikit = ridge_scikit.predict(ridge.X_test)
    # our method:
    ridge.regression_method()
    evalmodel_test = ridge.eval_model

    np.testing.assert_allclose(evalmodel_scikit, evalmodel_test, rtol=1e-5)


# -------------------- k-fold cross validation test --------------------------
def test_kfold_OLS():
    """
    Testing that our kfold implementation with OLS works the same as Scikit-Learn's
    within a relative tolerance 1e-2 (Higher tolerance because of stochasticity).
    """
    k = 5
    # our method:
    OLS_test = OrdinaryLeastSquares(stddev=0, n_points=100)
    EPE_test = OLS_test.k_fold(k)
    # scikit-learn method:
    kfold = sklms.KFold(n_splits=k, shuffle=True)
    OLS_scikit = skllm.LinearRegression()
    EPE_scikit = np.mean(
        -sklms.cross_val_score(
            OLS_scikit,
            OLS_test.X_train,
            OLS_test.z_train,
            scoring="neg_mean_squared_error",
            cv=kfold,
            n_jobs=-1,
        )
    )
    np.testing.assert_allclose(EPE_test, EPE_scikit, rtol=0.05)


# -------------------------- test of R2 score ------------------------------
def test_R2():
    """
    Testing that our implementation of the R2 score corresponds with 
    Scikit-learn's built-in R2 score of the prediction within a tolerance 1e-5
    """
    lambda_ = 0.1
    ridge = RidgeRegression(stddev=0, n_points=50, lambd=lambda_)
    # sciki-learn method:
    ridge_scikit = skllm.Ridge(alpha=lambda_).fit(ridge.X_train, ridge.z_train)
    R2_scikit = ridge_scikit.score(ridge.X_test, ridge.z_test)
    # our method: 
    ridge.regression_method() 

    np.testing.assert_allclose(ridge.r_squared, R2_scikit, rtol=1e-5)


# --------------------- test of the mean squared error ----------------------
def test_MSE():
    """
    Testing that our implementation of the mean squared error corresponds with 
    Scikit-learn's built-in function for MSE within a tolerance 1e-5
    """
    OLS = OrdinaryLeastSquares(stddev=0, n_points=50)
    OLS.regression_method() 
    MSE_scikit = skm.mean_squared_error(OLS.eval_model, OLS.z_test)
    np.testing.assert_allclose(OLS.mean_squared_error, MSE_scikit, rtol=1e-5)
    

