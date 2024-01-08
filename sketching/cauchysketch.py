import numpy as np
from scipy.stats import cauchy

class Cauchysketch:
    """Sketch used to calculate a CauchySketchingExperiment"""
    
    def __init__(self, N, n, d):
        self.N = N
        self.n = n
        self.d = d

        shape = (N, d)
        self.Z_prime = np.zeros(shape)
        self.weights = np.ones(N)

    def insert(self, x_vec):
        rand_cauchy = cauchy.rvs(size=self.N)
        self.Z_prime += np.outer(rand_cauchy, x_vec)

    def get_reduced_matrix(self):
        return self.Z_prime

    def get_weights(self):
        return self.weights
