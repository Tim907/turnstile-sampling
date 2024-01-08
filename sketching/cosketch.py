import math
import numpy as np
from numba import jit


@jit(nopython=True)
def _coinsert(X_prime, x_vec, alpha, b, N, h_max, s):
    x = np.random.randint(0, alpha)
    l = math.floor(np.log(x * (b - 1) + 1) / np.log(b))  # noqa: E741
    x = np.random.randint(0, N)
    if l < h_max-1 :
        elem = l * (N) + x
        X_prime[elem] = X_prime[elem] + x_vec

    N2 = int(N/s)
    for k in range(0, s):
        x = np.random.randint(0, N2)
        elem = (h_max-1) * (N) + k * N2 + x
        X_prime[elem] = X_prime[elem] + x_vec

    return X_prime


class Cosketch:
    """Sketch used to calculate an Oblivious Sketching Experiment when parameter cohensketch > 1"""
    
    def __init__(self, h_max, b, N, n, d, s):
        self.h_max = h_max
        self.b = b
        self.N = N
        self.n = n
        self.d = d
        self.s = s

        shape = (N * h_max, d)
        self.X_prime = np.zeros(shape)

        self.alpha = 0
        for l in range(0, h_max):  # noqa: E741
            self.alpha += b ** (l)

        self.p = np.repeat(0.0, h_max)
        self.w = np.repeat(0.0, h_max)
        for l in range(0, h_max-1):  # noqa: E741
            self.p[l] = 1 / (self.alpha * (b ** (-l)))
            self.w[l] = 1 / self.p[l]
        
        self.w[h_max-1] = 1 / self.s
        self.weights = np.repeat(0.0, N * h_max)
        for l in range(0, N * h_max):  # noqa: E741
            self.weights[l] = self.w[int(l / N)]

    def coinsert(self, x_vec):
        self.X_prime = _coinsert(self.X_prime, x_vec, self.alpha, self.b, self.N, self.h_max, self.s)

    def get_reduced_matrix(self):
        return self.X_prime

    def get_weights(self):
        return self.weights
