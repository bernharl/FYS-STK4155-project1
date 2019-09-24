import sys
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
sys.path.append('../')
from main import OrdinaryLeastSquares, RidgeRegression, LassoRegression



# --------------- k-fold cross validation test -----------------------
def test_kfold():
    k = 5
    kfold = sklms.KFold(n_splits=k, shuffle=True)
    OLS_test = OrdinaryLeastSquares()
    EPE_test = OLS_test.k_fold(k)
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
    relative = EPE_test/EPE_scikit
    np.testing.assert_almost_equal(relative, 1,  decimal=1)


# ------------ test of OLS regression parameters ------------------------
def test_OLS():
    pass