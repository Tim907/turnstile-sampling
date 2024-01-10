import abc
import logging
from time import perf_counter

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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

        Z = self.dataset.get_Z()

        beta_opt = self.dataset.get_beta_opt(self.optimizer)
        objective_function = self.optimizer.get_objective_function()
        f_opt = objective_function(beta_opt)

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
            h_max,
            kyfan_percent,
            sketchratio,
            cohensketch,
            optimizer: optimizer.base_optimizer,
            algorithm
    ):
        if algorithm not in {2, 3}:
            raise ValueError("Algorithm must be one of {2, 3}.")
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
            optimizer=optimizer,
        )
        self.algorithm = algorithm


    def get_reduced_matrix_and_weights(self, config):
        Z = self.optimizer.get_Z()
        n = self.dataset.get_n()
        d = Z.shape[1]
        size = config["size"]

        s = 5
        p = 1
        k = round(size * np.log(n) / 10)
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

        # turnstile stream updates
        for i in range(n):
            for j in range(s):
                B_j_list[j][h_i_j_mat[i, j], :] = B_j_list[j][h_i_j_mat[i, j], :] + sigma_i_j_mat[i, j] * Z[i, :] / t_i[
                    i] ** (1 / p)

        # Multiplication B_j = B_j * R
        f = np.random.randint(d ** 2, size=n)
        g = np.random.randint(2, size=n) * 2 - 1
        Z_ = np.zeros((d ** 2, d))
        for i in range(n):
            Z_[f[i]] += g[i] * Z[i]
        R_ = np.linalg.qr(Z_, mode="r")
        for j in range(s):
            B_j_list[j] = np.matmul(B_j_list[j], R_)

        a_i_mat = np.zeros((n, d))
        v_i = np.zeros(n)
        for i in range(n):
            a_i_j = np.zeros(s * d).reshape((s, d))
            for j in range(s):
                a_i_j[j, :] = sigma_i_j_mat[i, j] * B_j_list[j][h_i_j_mat[i, j]]

            v_i[i] = np.median(np.linalg.norm(a_i_j, ord=p, axis=1) ** p)
            # Find j minimizing median
            temp = np.zeros(s)
            for j in range(s):
                temp[j] = np.median(np.linalg.norm(a_i_j[j, :] - a_i_j, ord=p, axis=1) ** p) # median of s elements
            a_i_mat[i, :] = a_i_j[np.argmin(temp), :]

        # Select a_i with k largest v_i
        index_of_largest = np.argsort(-v_i)[0:k]
        reduced_matrix = a_i_mat[index_of_largest, :]
        t_i = t_i[index_of_largest]

        # calculate alpha
        norms = np.linalg.norm(reduced_matrix, ord=p, axis=1) # is of length n
        alpha = np.min(np.linalg.norm(reduced_matrix[np.argsort(norms)[0:k], :], ord=p, axis=1) ** p)

        reversed = reduced_matrix * t_i[:, np.newaxis] ** (1 / p)
        weights = (np.linalg.norm(reversed, ord=p, axis=1) ** p) / alpha
        weights[weights < 1] = 1

        # Theorem 4.1 https://arxiv.org/pdf/1801.04414.pdf
        reversed = reversed * d * np.log(d)
        R_1 = reversed.shape[0]
        R_2 = round(min(R_1, d ** 1.1))

        # Random map with uniform probability
        h = np.random.randint(0, R_2, n)

        Pi_2 = np.zeros((R_2, n))
        Pi_2[h, np.arange(n)] = np.random.standard_cauchy(n)

        #row_indices = _rng.choice(n, size=size, replace=False)
        #reduced_matrix = Z[row_indices]
        #weights = weights[row_indices]
        return reduced_matrix, weights

    def optimize(self, reduced_matrix, weights):
        return optimizer.optimize(
            reduced_matrix,
            weights,
            block_size=self.cur_kyfan_block_size,
            k=self.cur_kyfan_k,
            max_len=self.cur_kyfan_max_len,
        ).x