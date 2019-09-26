import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
from main import OrdinaryLeastSquares, RidgeRegression, LassoRegression

"""
Test of our OLS and Ridge regression methods from main.py, as well as the k-fold 
cross validation method. These are all tested using the artificial data from the
Franke function
"""

# -------------------- test of OLS regression -----------------------------
def test_OLS():
    OLS = OrdinaryLeastSquares()
    # scikit-learn method:
    OLS_scikit = skllm.LinearRegression().fit(OLS.X_train, OLS.z_train)
    evalmodel_scikit = OLS_scikit.predict(OLS.X_test)
    # our method:
    OLS.regression_method()
    evalmodel_test = OLS.eval_model

    np.testing.assert_allclose(evalmodel_scikit, evalmodel_test, rtol=1e-5)


# -------------------- test of Ridge regression --------------------------
def test_ridge():
    ridge = RidgeRegression()
    # scikit-learn method:
    ridge_scikit = skllm.Ridge(alpha=ridge.lambd).fit(ridge.X_train, ridge.z_train)
    evalmodel_scikit = ridge_scikit.predict(ridge.X_test)
    # our method:
    ridge.regression_method()
    evalmodel_test = ridge.eval_model

    np.testing.assert_allclose(evalmodel_scikit, evalmodel_test, rtol=1e-5)


# --------------- k-fold cross validation test -----------------------
def test_kfold_OLS():
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
