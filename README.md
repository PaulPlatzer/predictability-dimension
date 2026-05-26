# Predictability Dimension

Exploratory research code to compare local attractor dimension with atmospheric
predictability proxies.

The current workflow estimates local dimensions directly from raw ERA5 z500
fields on the 2-degree grid stored in `data/`, then compares those dimensions
with TIGGE ensemble spread.  Notebooks are kept for displaying and discussing
results; reusable computations live in Python modules under `src/`.

## Local Environment

Using `conda`:

```bash
conda env create -f environment.yml
conda activate predictability-dimension
python -m ipykernel install --user --name predictability-dimension
```

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks]"
```

The package provides two command-line entry points after installation:

```bash
compute-raw-dimension --help
compare-tigge-spread --help
```

The same commands can also be run through the scripts in `scripts/`.

## Data

Tracked data currently include:

- `data/ERA5_z500_1979_2023_subset_2deg.nc`: raw z500 ERA5 field used for the
  new dimension workflow.
- `data/ERA5_z500_1979_2023_subset_3deg.nc`, `4deg.nc`, `5deg.nc`: coarser
  versions kept for sensitivity checks.
- `data/TIGGEspread.nc`: TIGGE spread aligned later with dimension estimates.
- `data/dim.nc`: legacy dimension output produced from PCA-based pseudo-PCs.

Some older notebooks refer to `data/pcs.nc`, `data/eofs.nc`, and
`data/pourc_eofs.nc`. Those files are not part of the current repository and
belong to the previous PCA-based workflow.

## Main Workflow

Compute local dimensions from the raw 2-degree ERA5 field:

```bash
compute-raw-dimension \
  --input data/ERA5_z500_1979_2023_subset_2deg.nc \
  --output data/dim_raw_era5_2deg.nc \
  --k-values 10 20 50 100 \
  --knn-kmax 500 \
  --catalog-step 7 \
  --n-jobs 14
```

By default the raw field is transformed into state vectors using:

- temporal anomaly at each grid point,
- division by the temporal standard deviation at each grid point,
- `sqrt(cos(latitude))` area weighting.

These choices are visible command-line options and can be disabled with
`--no-anomaly`, `--no-standardize`, and `--no-area-weight`.

Compare dimension estimates with TIGGE spread:

```bash
compare-tigge-spread \
  --spread data/TIGGEspread.nc \
  --dimension data/dim_raw_era5_2deg.nc \
  --output-dir figures/generated
```

If a notebook still expects `data/dim.nc`, either point it to the new file or
write the raw-field dimensions directly to `data/dim.nc`.

## Code Structure

```text
src/predictability_dimension/
  fields.py       # open raw z500 fields and build sample x feature matrices
  analogues.py    # nearest-neighbour analogue search and analogue CRPS
  dimension.py    # local MLE, lPCA, and ESS dimension estimators
  scores.py       # climatological and horizon-wise CRPS utilities
  plotting.py     # plotting helpers used by notebooks and scripts
  preprocess.py   # legacy PCA-based preprocessing helper
  cli/            # command-line workflows

scripts/
  compute_raw_dimension.py
  compare_tigge_spread.py

utils/
  compatibility wrappers for older notebooks
```

The `utils/` modules are intentionally thin wrappers so existing notebooks do
not break immediately. New code should import from `predictability_dimension`.

## Notebooks

- `Dimension_vs_CRPS_horizons.ipynb`: legacy PCA-based CRPS/dimension
  exploration.
- `TIGGEspread_dim.ipynb`: display notebook comparing dimensions and TIGGE
  spread.
- `utils/CRPS_clim.ipynb`: exploratory checks for climatological CRPS.
- `notebooks/legacy/`: backup copies of the legacy notebooks.
- `notebooks/results/Raw_ERA5_dimension_TIGGE_display.ipynb`: clean display
  notebook for the raw ERA5 dimension workflow.

For collaborative work, keep heavy computations in the package/scripts and use
notebooks mainly to load NetCDF outputs, make figures, and record scientific
interpretation.

## Quick Smoke Run

The full dimension computation can be expensive. To check that the environment
and pipeline are working before launching a production run:

```bash
python scripts/compute_raw_dimension.py \
  --output outputs/smoke/dim_raw_era5_2deg_all_estimators_smoke.nc \
  --k-values 5 8 \
  --knn-kmax 10 \
  --catalog-step 1 \
  --sample-step 500 \
  --estimators mle pca ess \
  --n-jobs 2

python scripts/compare_tigge_spread.py \
  --dimension outputs/smoke/dim_raw_era5_2deg_all_estimators_smoke.nc \
  --output-dir outputs/smoke/figures_all_estimators
```
