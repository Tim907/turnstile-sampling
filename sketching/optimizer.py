import logging

import numpy as np
import scipy.optimize as so
from numba import jit
from sklearn.linear_model import SGDClassifier
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model import LogisticRegression


def only_keep_k(vec, block_size, k, max_len=None, biggest=True):
    """
    Only keep the k biggest (smalles) elements for each block in a vector.

    If max_len = None, use the whole vec. Otherwise, use vec[:max_len]

    Returns: new vector, indices
    """

    if k == block_size:
        return vec, np.array(list(range(len(vec))))

    do_not_touch = np.array([])
    if max_len is not None:
        do_not_touch = vec[max_len:]
        vec = vec[:max_len]

    # determine the number of blocks
    num_blocks = int(vec.shape[0] / block_size)

    # split the vector in a list of blocks (chunks)
    chunks = np.array_split(vec, num_blocks)

    # chunks_new will contain the k biggest (smallest) elements for each chunk
    chunks_new = []
    keep_indices = []
    for i, cur_chunk in enumerate(chunks):
        if biggest:
            cur_partition_indices = np.argpartition(-cur_chunk, k)
        else:
            cur_partition_indices = np.argpartition(cur_chunk, k)
        chunks_new.append(cur_chunk[cur_partition_indices[:k]])
        keep_indices.extend(cur_partition_indices[:k] + i * block_size)

    if max_len is not None:
        chunks_new.append(do_not_touch)
        keep_indices.extend(
            list(range(vec.shape[0], vec.shape[0] + do_not_touch.shape[0]))
        )

    return np.concatenate(chunks_new), np.array(keep_indices)


@jit(nopython=True)
def calc(v):
    """calculates log(1 + exp(v)), but becomes identity function if v is bigger than 34"""
    if v < 34:
        # prevent underflow exception
        if(v < -200): 
            return np.exp(-200)

        return np.log1p(np.exp(v))
    return v
        

calc_vectorized = np.vectorize(calc)


def logistic_likelihood(theta, Z, weights=None, block_size=None, k=None, max_len=None):
    v = -Z.dot(theta)
    if block_size is not None and k is not None:
        v, indices = only_keep_k(v, block_size, k, max_len=max_len, biggest=True)
        if weights is not None:
            weights = weights[indices]
    likelihoods = calc_vectorized(v)
    if weights is not None:
        likelihoods = weights * likelihoods.T
    return np.sum(likelihoods)


def logistic_likelihood_grad(
    theta, Z, weights=None, block_size=None, k=None, max_len=None):
    v = Z.dot(theta)
    if block_size is not None and k is not None:
        v, indices = only_keep_k(v, block_size, k, max_len=max_len, biggest=False)
        if weights is not None:
            weights = weights[indices]
        Z = Z[indices, :]

    grad_weights = 1.0 / (1.0 + np.exp(v))

    if weights is not None:
        grad_weights *= weights

    return -1 * (grad_weights.dot(Z))


def L1_objective(theta, X, y):
    """L1 loss function"""
    return np.sum(np.abs(X.dot(theta) - y))

def L1_grad(theta, X, y):
    """L1 gradient function"""
    return np.sum(np.multiply(X, np.sign(X.dot(theta) - y)[:, np.newaxis]), axis=0)


def optimize(Z, w=None, block_size=None, k=None, max_len=None):

    if w is None:
        w = np.ones(Z.shape[0])

    def objective_function(theta):
        return logistic_likelihood(theta, Z, w, block_size=block_size, k=k, max_len=max_len)

    def gradient(theta):
        return logistic_likelihood_grad(theta, Z, w, block_size=block_size, k=k, max_len=max_len)

    theta0 = np.zeros(Z.shape[1])

    res = so.minimize(objective_function, theta0, method="L-BFGS-B", jac=gradient)
    if res.success is False:
        print("Optimization not successful.")
        print(res)
    return res


def optimize_L1(Z):
    """Optimizes by L1 loss according to Theorem 1 of the paper."""

    X = Z[:, 0:(Z.shape[1]-1)]
    y = Z[:, -1]

    def objective_function(theta):
        return L1_objective(theta, X, y)

    def gradient(theta):
        return L1_grad(theta, X, y)

    theta0 = np.zeros(X.shape[1])

    # results = []
    # for method in ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
    #                 'l-bfgs-b', 'tnc', 'cobyla', 'slsqp', 'trust-constr']:
    #     results.append(so.minimize(objective_function, theta0, method=method, jac=gradient))

    theta0 = np.random.uniform(size = X.shape[1])
    res = so.minimize(objective_function, theta0, method="L-BFGS-B", jac=gradient)
    if res.success is False:
        print("Optimization not successful.")
        print(res)
    return res


class base_optimizer:
    """optimizer for logistic regression"""

    def __init__(self) -> None:
        return

    def get_name(self):
        return "logistic"

    def setDataset(self, X, y, Z):
        self.X = X
        self.y = y
        self.Z = Z

    def get_Z(self):
        return self.Z

    def optimize(self, reduced_matrix, weights=None):
        return optimize(Z = reduced_matrix, w = weights).x

    def get_objective_function(self):
        return lambda theta: logistic_likelihood(theta, self.Z)


class L1_optimizer(base_optimizer):
    """optimizer for L1 optimization"""

    def get_name(self):
        return "L1"

    def optimize(self, reduced_matrix, weights=None):
        return optimize_L1(reduced_matrix).x

    def get_objective_function(self):
        Z = self.get_Z()
        return lambda theta: L1_objective(theta, X=Z[:, 0:(Z.shape[1]-1)], y=Z[:, -1])

    def get_Z(self):
        return np.append(np.append(self.X, np.ones(shape=(self.X.shape[0], 1)), axis=1), self.y[:, np.newaxis], axis=1)
