# Leveraging Argument Structure to Predict Content Hatefulness

This repository contains the code accompanying the paper **"Leveraging Argument Structure to Predict Content Hatefulness"**.

## TLDR

This repository includes:

- The code used to reproduce the experiments reported in the paper, located in `experiments/`.
  - `01_data_stats.py` reproduces the dataset statistics reported in **Table 1**.
  - `02_predict.py` reproduces the classification results reported in **Table 2**.
- The `data/` directory contains a copy of the **WSF-ARG+** dataset extracted from the related work.

## Installation

Create a Python environment using your preferred environment manager. The experiments reported in the paper were run using `venv`.

```bash
python -m venv arg_hs_detection_venv
source arg_hs_detection_venv/bin/activate
pip install -r requirements.txt
```

## Running the Experiments

Activate the virtual environment and execute the Python scripts in the `experiments/` directory using the IDE of your choice.