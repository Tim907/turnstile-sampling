from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# the downloaded datasets will go here
DATA_DIR = BASE_DIR / ".data-cache"

# the results of run_experiments will go here
RESULTS_DIR = BASE_DIR / "experimental-results"

# jupyter notebook plots will be automatically exported here
PLOTS_DIR = BASE_DIR / "plots"

LOGGER_NAME = "sketching"
