from sketching.datasets import Synthetic_Dataset_Cohen
from sketching.utils import run_experiments

"""
HOW TO:
Min_size, max_size and step_size leads to the grid of different sketch sizes used.
Num_runs defines the number of replications. Quantities for the plots are calculated 
with the median of all replications, so better use uneven numbers.

Below you can change the variance-regularization hyperparameter.
Inside the function utils.run_experiments, different optimizers can be configured from optimizer.py: 
logistic likelihood, variance-regularized logistic likelihood, L1-optimization and SGD
"""

MIN_SIZE = 1000
MAX_SIZE = 4000
STEP_SIZE = 250
NUM_RUNS = 31

dataset = Synthetic_Dataset_Cohen(n_rows=20000,d_cols=100)

run_experiments(
    dataset=dataset,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS
)
