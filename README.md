# Atlantic water-mass classification (OMP → RF → ECCO)

If you use this software, please cite the archived Zenodo release.

Code accompanying the manuscript (submitted to *JGR: Machine Learning and Computation*) describing a two-stage framework:
1) an OMP-derived, tracer-informed reference classification, and
2) a Random Forest (RF) ensemble trained to reproduce OMP water-mass fractions from T/S + position, then applied to ECCO to generate gridded monthly Atlantic water-mass fields.

## Quick start (minimal workflow)

1. (Optional) Run OMP notebook to generate training CSV:
   notebooks/atlantic_omp.ipynb

2. Generate ECCO height-above-bottom (HAB) file:
   python scripts/make_ecco_bottom_depth_3d.py --input <ECCO_file> --output ecco_bottom_depth_3d.npy

3. Train + infer:
   python scripts/batch_inference_own_omp.py <ECCO_ROOT_DIR> <PRED_DIR|same> true 16

## What’s in this repo

### Stage 1: OMP (reference classification)
OMP is run in a notebook (`notebooks/atlantic_omp.ipynb`) to generate a training dataset of water-mass fractions.

If you already have OMP output from another pipeline, you can skip this notebook and supply your own training table, but you must ensure the later stages (feature names/columns) match what the ML scripts expect (see below).

### Stage 2–3: ML training + batch inference to ECCO (main entrypoint)
The main script is:

- `scripts/batch_inference_own_omp.py`

This script can:
- train the RF ensemble only,
- run inference only (using pre-trained models), or
- do both (train then infer).

Under the hood it uses `scripts/trainKfold.py` for 5-fold RF training and ensemble inference.

### Validation + visualisation
- `notebooks/ML_verification_atlantic.ipynb` contains model validation experiments (including spatial / out-of-distribution testing as described in the manuscript).
- `notebooks/visualising_output.ipynb` contains plotting/visualisation of the gridded ECCO outputs.

## Data required (not included)

You must download the underlying datasets separately:
- GLODAP merged and adjusted product (used to generate the OMP reference classification / training labels)
- ECCO v4r4 (used for gridded inference)

## Expected training table format

The batch script trains from a CSV of OMP-labelled samples (water-mass fractions), with the following input feature columns:

- `conservative_temperature`
- `abs_salinity`
- `latitude`
- `longitude_sin`
- `longitude_cos`
- `pressure`
- `hab` (height above bottom)

and the following output columns (water-mass fractions):

- `CW`, `AAIW`, `SAIW`, `uNADW`, `lNADW`, `CDW`, `AABW`

If you use a different OMP workflow, you can still use this repo, but you will need to adjust the column names and/or feature construction accordingly. You may also choose to include additional training features. 

## Running the batch script

### 0) Prepare inputs
1. Put your ECCO NetCDF files in a directory (the script searches recursively).
2. Ensure you have either:
   - a training CSV produced by the OMP notebook, or
   - pre-trained RF models saved in `./models/`

3. Generate height-above-bottom (HAB) file required by the batch script:

   python scripts/make_ecco_bottom_depth_3d.py \
       --input <ECCO_sample_file.nc> \
       --output ecco_bottom_depth_3d.npy

   This file must be present in the working directory when running the batch script.

**Important:** the current batch script contains a hard-coded path to the training CSV; you will need to edit that line to point to your local CSV before training.

### 1) Train + infer (recommended “one command” workflow)
```bash
python scripts/batch_inference_own_omp.py <ECCO_ROOT_DIR> <PRED_DIR|same> true 16
```
Arguments:

<ECCO_ROOT_DIR>: root directory containing ECCO NetCDF files

<PRED_DIR|same>:
  - If a directory path: predictions are written there.
  - If "same": predictions are written alongside each input ECCO file.


true: train models before inference

16: RF max depth (default 16)

Models are written to ./models/, and are saved as:
./models/RF_depth{depth}_fold{0–4}.joblib

### 2) Inference only (use existing models)
python scripts/batch_inference_own_omp.py <ECCO_ROOT_DIR> <PRED_DIR|same> false 16

This loads model files matching ./models/RF_depth<depth>_fold*.

### Outputs

For each ECCO time slice processed, the batch script writes:

a NetCDF containing the predicted water-mass fractions (and copies theta, salt, latitude, longitude, depth)

a companion NetCDF containing uncertainty estimates (ensemble variance) for each water mass

### Python environment

This repo is standard scientific Python. You will need (at minimum):

numpy, pandas

scikit-learn

netCDF4

gsw

joblib

matplotlib

(An environment.yml / requirements.txt can be added once the repo is stable.)

### Notes

The batch inference script converts ECCO potential temperature and practical salinity to Conservative Temperature and Absolute Salinity using TEOS-10 (gsw).

The training/inference features are currently “thermohaline + positional”; tracer features exist in the codebase but are not used in the default setting.
