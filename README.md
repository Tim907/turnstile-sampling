# Turnstile Sampling

[![python-version](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)

This is the accompanying code repository for the **ICML 2024** publication **"Turnstile ℓp leverage score sampling with applications"** by Alexander Munteanu and Simon Omlor, implementation and experiments supported by Tim Novak.

## How to install

1. Clone the repository and navigate into the new directory

   ```bash
   git clone https://github.com/Tim907/turnstile-sampling
   cd turnstile-sampling
   ```

2. Create and activate a new virtual environment
   
   on Unix:
   ```bash
   python -m venv venv
   . ./venv/bin/activate
   ```
   on Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate.bat
   ```

3. Install the package locally

   ```bash
   pip install .
   ```

4. To confirm that everything worked, install `pytest` and run the tests
   ```bash
   pip install pytest
   python -m pytest
   ```

## How to run the experiments

The `scripts` directory contains multiple python scripts that can be
used to run the experiments.
Just make sure, that everything is installed properly.

For example, to run the covertype experiments you can use the following command:

```bash
python scripts/run_experiments_covertype.py
```

## How to recreate the plots

The plots can be recreated using the jupyter notebooks that can be
found in the `notebooks` directory.
Instructions on how to set up a jupyter environment can be found
[here](https://jupyter.org/).
