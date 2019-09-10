import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
from main.py import OrdinaryLeastSquares, RidgeRegression, LassoRegression

def test_kfold():
    OLS_test = OrdinaryLeastSquares()
    EPE = OLS_test.k_fold()
