import abc
import logging
from time import perf_counter

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import expon

from . import optimizer, settings
from .datasets import Dataset
from .l2s_sampling import l2s_sampling
from .sketch import Sketch
from .cosketch import Cosketch
from .cauchysketch import Cauchysketch

logger = logging.getLogger(settings.LOGGER_NAME)

_rng = np.random.default_rng()


class BaseExperiment(abc.ABC):
    def __init__(
        self,
        num_runs,
        min_size,
        max_size,
        step_size,
        dataset: Dataset,
        results_filename,
        optimizer: optimizer.base_optimizer,
    ):
        self.num_runs = num_runs
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size
        self.dataset = dataset
        self.results_filename = results_filename
        self.optimizer = optimizer
        self.optimizer.setDataset(dataset.get_X(), dataset.get_y(), dataset.get_Z())

    @abc.abstractmethod
    def get_reduced_matrix_and_weights(self, config):
        pass

    def get_config_grid(self):
        """
        Returns a list of configurations that are used to run the experiments.
        """
        grid = []
        for size in np.arange(
            start=self.min_size,
            stop=self.max_size + self.step_size,
            step=self.step_size,
        ):
            for run in range(1, self.num_runs + 1):
                grid.append({"run": run, "size": size})

        return grid

    def optimize(self, reduced_matrix, weights):
        return optimizer.optimize(Z=reduced_matrix, w=weights).x

    def run(self, parallel=False, n_jobs=-2, add=False):
        """Runs the experiment with given settings. Can take a few minutes.

        Parameters
        ----------
        parallel : bool
            A flag used if multiple CPU Cores should be used to run different sketch sizes of the grid in parallel.
        n_jobs : int
            The number of CPU cores used. If -1 all are used. For n_jobs = -2, all but one are used. For n_jobs = -3, all but two etc.
        add : bool, optional
            A flag used if the experimental result should be appended to the .csv (True) otherwise overwrite with new .csv (False).
            Useful if one wants to calculate more replications afterwards for a smoother plot.
        """

        beta_opt = self.dataset.get_beta_opt(self.optimizer)
        objective_function = self.optimizer.get_objective_function()
        f_opt = objective_function(beta_opt)
        logger.info(f"optimal cost function: {f_opt}")
        logger.info("Running experiments...")

        def job_function(cur_config):
            logger.info(f"Current experimental config: {cur_config}")

            start_time = perf_counter()

            reduced_matrix, weights = self.get_reduced_matrix_and_weights(cur_config)
            sampling_time = perf_counter() - start_time

            cur_beta_opt = self.optimizer.optimize(reduced_matrix, weights)
            total_time = perf_counter() - start_time

            cur_ratio = objective_function(cur_beta_opt) / f_opt
            return {
                **cur_config,
                "ratio": cur_ratio,
                "sampling_time_s": sampling_time,
                "total_time_s": total_time,
            }

        if parallel:
            results = Parallel(n_jobs=n_jobs)(
                delayed(job_function)(cur_config)
                for cur_config in self.get_config_grid()
            )
        else:
            results = [
                job_function(cur_config) for cur_config in self.get_config_grid()
            ]

        logger.info(f"Writing results to {self.results_filename}")

        df = pd.DataFrame(results)
        if not os.path.isfile(self.results_filename) or add is False:
            df.to_csv(self.results_filename, index=False)
        else:
            df.to_csv(self.results_filename, mode="a", header=False, index=False)

        logger.info("Done.")


class UniformSamplingExperiment(BaseExperiment):
    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs,
        optimizer: optimizer.base_optimizer,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
            optimizer=optimizer,
        )

    def get_reduced_matrix_and_weights(self, config):
        Z = self.dataset.get_Z()
        n = self.dataset.get_n()
        size = config["size"]

        row_indices = _rng.choice(n, size=size, replace=False)
        reduced_matrix = Z[row_indices]
        weights = np.ones(size)

        return reduced_matrix, weights


class CauchySketchingExperiment(BaseExperiment):
    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs,
        optimizer: optimizer.base_optimizer,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
            optimizer=optimizer
        )

    def get_reduced_matrix_and_weights(self, config):

        Z = self.optimizer.get_Z()
        n = self.dataset.get_n()
        size = config["size"]
        d = Z.shape[1]

        sketch = Cauchysketch(size, n, d)
        for j in range(0, n):
            sketch.insert(Z[j])

        reduced_matrix = sketch.get_reduced_matrix()
        weights_sketch = sketch.get_weights()

        return reduced_matrix, weights_sketch


class ObliviousSketchingExperiment(BaseExperiment):
    """
    WARNING: This implementation is not thread safe!!!
    """

    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs,
        h_max,
        kyfan_percent,
        sketchratio,
        cohensketch,
        optimizer: optimizer.base_optimizer,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
            optimizer = optimizer,
        )
        self.h_max = h_max
        self.kyfan_percent = kyfan_percent
        self.sketchratio = sketchratio
        self.cohensketch = cohensketch

    def get_reduced_matrix_and_weights(self, config):
        Z = self.optimizer.get_Z()
        n = self.dataset.get_n()
        d = Z.shape[1]
        size = config["size"]

        # divide by (h_max + 1) + to get one more block for unif sampling
        if self.cohensketch > 1 :
            N2 = max(int(size * self.sketchratio / (self.h_max * self.cohensketch)), 1)
            N = N2 * self.cohensketch
            b = (n / N) ** (1.0 / self.h_max)
            actual_sketch_size = N * self.h_max
            
            unif_block_size = max(size - actual_sketch_size, 1)
            
            sketch = Cosketch(self.h_max, b, N, n, d, self.cohensketch)
            for j in range(0, n):
                sketch.coinsert(Z[j])
            reduced_matrix = sketch.get_reduced_matrix()
            weights_sketch = sketch.get_weights()
        else:
            N = max(int(size * self.sketchratio / self.h_max), 1)
            b = (n / N) ** (1.0 / self.h_max)
            actual_sketch_size = N * self.h_max
            
            unif_block_size = max(size - actual_sketch_size, 1)
            
            sketch = Sketch(self.h_max, b, N, n, d)
            for j in range(0, n):
                sketch.insert(Z[j])
            reduced_matrix = sketch.get_reduced_matrix()
            weights_sketch = sketch.get_weights()

        # do the unif sampling
        rows = _rng.choice(n, unif_block_size, replace=False)
        unif_sample = Z[rows]

        # concat the sketch and the uniform sample
        reduced_matrix = np.vstack([reduced_matrix, unif_sample])

        weights_unif = np.ones(unif_block_size) * n / unif_block_size

        weights = np.concatenate([weights_sketch, weights_unif])
        weights = weights / np.sum(weights)

        self.cur_kyfan_k = int(N * self.kyfan_percent)
        self.cur_kyfan_max_len = actual_sketch_size
        self.cur_kyfan_block_size = N

        return reduced_matrix, weights

    def optimize(self, reduced_matrix, weights):
        return optimizer.optimize(
            reduced_matrix,
            weights,
            block_size=self.cur_kyfan_block_size,
            k=self.cur_kyfan_k,
            max_len=self.cur_kyfan_max_len,
        ).x



class TurnstileSamplingExperiment(BaseExperiment):
    """
    WARNING: This implementation is not thread safe!!!
    """

    def __init__(
            self,
            dataset: Dataset,
            results_filename,
            min_size,
            max_size,
            step_size,
            num_runs,
            optimizer: optimizer.base_optimizer,
            factor_unif
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
            optimizer=optimizer,
        )
        self.factor_unif = factor_unif


    def get_reduced_matrix_and_weights(self, config):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        
        Z = self.optimizer.get_Z()
        n = self.dataset.get_n()
        d = Z.shape[1]
        
        k_unif = round(self.factor_unif * config["size"])  # uniform samples
        k = config["size"] - k_unif  # remaining samples of the sketch
        size = round( k * max(30, np.log(n) ) )
        s = 2 * round( max(5, np.log(n)/2 ) ) + 1
        p = 1

        print("Unif: "+str(self.factor_unif))
        print("Size (r): "+str(size))
        print("Size (s): "+str(s))
        
        if s / 2 == s // 2:
            raise ValueError("S should be an odd number.")

        # zero indexed hash maps of size n x s
        h_i_j_mat = np.random.randint(0, size, n*s).reshape((n, s))
        # mapping function to determine the sign {-1,1}
        sigma_i_j_mat = (np.random.randint(2, size=n*s) * 2 - 1).reshape((n, s))
        t_i = np.random.uniform(size=n)

        # Initialize B as list of s 0-matrices
        B_j_list = []
        for j in range(s):
            B_j_list.append(np.zeros((size, d)))

        # turnstile stream updates for Algo 1 & 2
        for i in range(n):
            for j in range(s):
                B_j_list[j][h_i_j_mat[i, j], :] += sigma_i_j_mat[i, j] * Z[i, :] / t_i[i] ** (1 / p)

        # turnstile stream updates for Algo 3
        f = np.random.randint((d ** 2), size=n)
        g = (np.random.randint(2, size=n) * 2 - 1) # * d
        lamb = expon.rvs(size=n)
        #f2 = np.random.randint(d ** 2, size=n)
        #g2 = np.random.standard_cauchy(n) / np.log(d)
        Z_ = np.zeros(((d**2),d))
        
        # QR decomposition for Algo3
        #for i in range(n):
        #    Z_[f[i]] += g[i] * Z[i, :] # Pi1*Z
        #    Z_[(d ** 2) + f2[i]] += g2[i] * Z[i, :] # Pi2*Z
        for i in range(n):
            Z_[f[i]] += g[i] * Z[i, :] / np.power(lamb[1],1/p) # Pi*Z
        R_ = np.linalg.qr(Z_, mode="r")
        R_inv = np.linalg.pinv(R_)
        
        # Post-Multiplication of sketches B_j = B_j * R
        for j in range(s):
            B_j_list[j] = np.matmul(B_j_list[j], R_inv)

        # ----Trivial implementation but much slower
        # a_i_mat_old = np.zeros((n, d))
        # v_i_old = np.zeros(n)
        # for i in range(n):
        #     a_i_j = np.zeros(s * d).reshape((s, d))
        #     for j in range(s):
        #         a_i_j[j, :] = sigma_i_j_mat[i, j] * B_j_list[j][h_i_j_mat[i, j]]
        #
        #     temp = np.linalg.norm(a_i_j, ord=p, axis=1) ** p
        #     ind = np.argsort(temp)
        #     index_of_median = ind[len(ind) // 2]
        #     v_i_old[i] = temp[index_of_median]
        #     a_i_mat_old[i, :] = a_i_j[index_of_median, :]

        # Compute a_i_j cube using vectorized operations
        a_i_j = np.stack([sigma_i_j_mat[:, j, np.newaxis] * B_j_list[j][h_i_j_mat[:, j]] for j in range(s)], axis=-1)
        # Compute the norm along axis j and raise it to the power of p
        temp = np.linalg.norm(a_i_j, ord=p, axis=1) ** p
        # Find the index of the median along axis j
        ind = np.argpartition(temp, s // 2, axis=1)[:, s // 2]
        v_i = temp[np.arange(n), ind]
        a_i_mat = a_i_j[np.arange(n), :, ind]

        # filter out unimportant samples
        number_filtered = 2 * k
        index_of_largest = np.argsort(-v_i)[0:number_filtered]
        outlier = np.zeros((number_filtered, s))
        for j in range(s):
            outlier[:, j] = np.linalg.norm(a_i_j[index_of_largest, :, ind[index_of_largest]] - a_i_j[index_of_largest, :, j], ord=p, axis=1) > 1 / 4 * v_i[index_of_largest]
        v_i[index_of_largest[np.sum(outlier, axis=1) >= s // 2]] = 0

        # Test equality of implementation
        # print(np.isclose(v_i, v_i_old).all())
        # print(np.isclose(a_i_mat, a_i_mat_old).all())
        # -> is equal

        # Select a_i with k largest v_i
        index_of_largest = np.argsort(-v_i)[0:k]
        reduced_matrix = a_i_mat[index_of_largest, :]
        t_i = t_i[index_of_largest]

        # calculate alpha
        norms = np.linalg.norm(reduced_matrix, ord=p, axis=1) # is of length n
        alpha = np.min(np.linalg.norm(reduced_matrix[np.argsort(norms)[0:k], :], ord=p, axis=1) ** p)

        reversed = reduced_matrix * t_i[:, np.newaxis] ** (1 / p)
        weights = (np.linalg.norm(reversed, ord=p, axis=1) ** p) / alpha + k_unif / n
        weights[weights > 1] = 1
        weights = 1/weights
        reduced_matrix = np.matmul(reversed, R_)

        # uniform sampling of k_unif
        row_indices = np.random.choice(n, size=k_unif, replace=False)
        reduced_matrix = np.vstack((reduced_matrix, Z[row_indices, :]))
        samples = np.matmul(Z[row_indices, :], R_inv)
        norms_unif = np.linalg.norm(samples/alpha, ord=p, axis=1) ** p + k_unif/n
        norms_unif[norms_unif > 1] = 1
        norms_unif = 1/norms_unif
        weights = np.concatenate((weights, norms_unif))

        # print("\nreduced_matrix distribution:\n")
        # print(pd.Series(reduced_matrix[:,0]).describe())
        # print("\nweights distribution:\n")
        # print(pd.Series(weights).describe())
        # print(weights)
        # print(n)
        # print(np.sum(weights))
        return reduced_matrix, weights

    def optimize(self, reduced_matrix, weights):
        return optimizer.optimize(
            reduced_matrix,
            weights,
            block_size=self.cur_kyfan_block_size,
            k=self.cur_kyfan_k,
            max_len=self.cur_kyfan_max_len,
        ).x



class LeverageScoreSamplingExperiment(BaseExperiment):
    """
    https://github.com/chr-peters/efficient-probit-regression/blob/cf5da81415f0b866a5971f38e92fa3d32e752c96/efficient_probit_regression/sampling.py#L243
    """

    def __init__(
            self,
            dataset: Dataset,
            results_filename,
            min_size,
            max_size,
            step_size,
            num_runs,
            optimizer: optimizer.base_optimizer,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
            optimizer=optimizer,
        )

    def fast_QR(self, X, p=1):
        """
        Returns Q of a fast QR decomposition of X.
        """
        n, d = X.shape

        if p <= 2:
            sketch_size = d ** 2
        else:
            sketch_size = np.maximum(d ** 2, int(np.power(n, 1 - 2 / p)))

        f = np.random.randint(sketch_size, size=n)
        g = np.random.randint(2, size=n) * 2 - 1
        if p != 2:
            lamb = expon.rvs(size=n)

        # init the sketch
        X_sketch = np.zeros((sketch_size, d))
        if p == 2:
            for i in range(n):
                X_sketch[f[i]] += g[i] * X[i]
        else:
            for i in range(n):
                X_sketch[f[i]] += g[i] / np.power(lamb[i], 1 / p) * X[i]  # exponential distributed random variable

        R = np.linalg.qr(X_sketch, mode="r")
        R_inv = np.linalg.inv(R)

        if p == 2:
            k = 20
            g = np.random.normal(loc=0, scale=1 / np.sqrt(k), size=(R_inv.shape[1], k))
            r = np.dot(R_inv, g)
            Q = np.dot(X, r)
        else:
            Q = np.dot(X, R_inv)
        return Q

    def compute_leverage_scores(self, X: np.ndarray, p, fast_approx):
        """
            Computes leverage scores.
        """
        if not len(X.shape) == 2:
            raise ValueError("X must be 2D!")

        if not fast_approx:  # boolean, fast or usual Q-R-decomposition
            Q, *_ = np.linalg.qr(X)
        else:
            Q = self.fast_QR(X, p=p)

        leverage_scores = np.power(np.linalg.norm(Q, axis=1, ord=p), p)

        return leverage_scores

    def get_reduced_matrix_and_weights(self, config):
        Z = self.optimizer.get_Z()
        size = config["size"]

        leverage_scores = self.compute_leverage_scores(Z, p=1, fast_approx=True)

        leverage_scores = leverage_scores / np.sum(leverage_scores)
        # augmented
        leverage_scores = leverage_scores + 0.2 / Z.shape[0]

        # calculate probabilities
        prob = leverage_scores / np.sum(leverage_scores)
        weights = 1 / (prob * size)
        weights[weights < 1] = 1
        sample_indices = np.random.choice(Z.shape[0], size=size, replace=False, p=prob)

        return Z[sample_indices, :], weights[sample_indices]