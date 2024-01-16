import numpy as np

from sketching.datasets import Covertype_Sklearn, All_Ones
from sketching.utils import run_experiments

MIN_SIZE = 20
MAX_SIZE = 20
STEP_SIZE = 1
NUM_RUNS = 1

dataset = All_Ones(n_rows=100000, d_cols=1, use_caching=False)

run_experiments(
    dataset=dataset,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
    add=False
)
