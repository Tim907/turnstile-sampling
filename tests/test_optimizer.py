import numpy as np
from numpy.testing import assert_allclose
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sketching import optimizer
from sketching.optimizer import base_optimizer
from scipy.stats import logistic


def test_iris():
    X, y = load_iris(return_X_y=True)
    y = np.where(y == 1, 1, -1)

    model = LogisticRegression(penalty="none", fit_intercept=False)
    model.fit(X, y)
    theta_opt = model.coef_[0]

    Z = np.multiply(y[:, np.newaxis], X)
    theta_optimizer = optimizer.optimize(Z, w=np.ones(y.shape)).x
    assert_allclose(theta_optimizer, theta_opt)


def test_iris_kyfan():
    X, y = load_iris(return_X_y=True)
    y = np.where(y == 1, 1, -1)

    model = LogisticRegression(penalty="none", fit_intercept=False)
    model.fit(X, y)
    theta_opt = model.coef_[0]

    Z = np.multiply(y[:, np.newaxis], X)
    theta_optimizer = optimizer.optimize(
        Z, w=np.ones(y.shape), block_size=10, k=10, max_len=y.shape[0]
    ).x

    assert_allclose(theta_optimizer, theta_opt)



def test_objective_function():
    X, y = load_iris(return_X_y=True)
    y = np.where(y == 1, 1, -1)
    Z = np.multiply(y[:, np.newaxis], X)

    theta = np.ones(X.shape[1])
    optim = base_optimizer()
    optim.setDataset(X, y, Z)
    objective_function = optim.get_objective_function()
    loss_optimizer = objective_function(theta)

    y_pred = np.matmul(X, theta)
    y_pred = logistic.cdf(y_pred)
    loss = log_loss(y, y_pred, normalize=False)

    assert_allclose(loss_optimizer, loss)
