import logging
from . import settings
from . import optimizer
from .datasets import Dataset
from .experiments import (
    CauchySketchingExperiment,
    ObliviousSketchingExperiment,
    TurnstileSamplingExperiment,
    UniformSamplingExperiment,
    LeverageScoreSamplingExperiment
)

logger = logging.getLogger(settings.LOGGER_NAME)

# Configure logging to write to file
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler("sketching.log")
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s","%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

def run_experiments(dataset: Dataset, min_size, max_size, step_size, num_runs, add=False):
    """Runs the sketching experiment and creates .csv output files in the experimental-results folder, used for the Jupyter Notebook plots.

    Parameters
    ----------
    dataset : Dataset
        The dataset to use for the experiment and Sketch
    min_size : int
        lower bound of the grid. First sketch size
    max_size : int
        upper bound of the grid. Not necessarily a valid sketch size used.
        Can possibly exceed max_size a single time, in case min_size + [multiple of step_size] < max_size.
    step_size : int
        interval of the grid
    num_runs : int
        Number of replications. Note that a median is taken of useful quantities, so try uneven numbers.
    add : bool
        A flag used if the experimental result should be appended to the .csv (True) otherwise overwrite with new .csv (False).
        Useful if one wants to calculate more replications afterwards for a smoother plot.
    """

    # check if results directory exists
    if not settings.RESULTS_DIR.exists():
        settings.RESULTS_DIR.mkdir()

    # logger.info("Starting leverage sampling experiment")
    experiment_leverage = LeverageScoreSamplingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_leverage.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        optimizer=optimizer.base_optimizer(),
    )
    #experiment_leverage.run(parallel=True, n_jobs=4, add=add)

    logger.info("Starting turnstile experiment")
    experiment_sketching = TurnstileSamplingExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_turnstile.csv",
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        optimizer=optimizer.base_optimizer(),
        factor_unif=0.2
    )
    experiment_sketching.run(parallel=True, n_jobs=4, add=add)

