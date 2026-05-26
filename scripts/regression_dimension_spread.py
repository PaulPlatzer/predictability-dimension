"""Regression analysis: TIGGE spread growth ~ dimension + density proxy.

The model fitted for each lead time h is:

    spread(t+h) / spread(t,0) = a(h) * x1(t) + b(h) * x2(t) + intercept(h)

where:
    x1 = local attractor dimension estimate from ERA5 (one estimator at a time)
    x2 = density proxy = mean distance to K=50 nearest analogues

Predictors are standardised before fitting so that coefficients a and b are
directly comparable.  Results are printed and saved as figures under
outputs/figures/regression_*.

Usage
-----
    python scripts/regression_dimension_spread.py
    python scripts/regression_dimension_spread.py --k-dim 50 --estimator d_mle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from predictability_dimension.config import DATA_DIR, DEFAULT_DIMENSION_OUTPUT, DEFAULT_TIGGE_SPREAD
from predictability_dimension.plotting import plot_regression_coefficients
from predictability_dimension.regression import fit_horizon_regressions

DEFAULT_ANA    = DATA_DIR / "analogue_spread_density.nc"
DEFAULT_OUTDIR = Path("outputs/figures")
HORIZONS_DAYS  = list(range(1, 8))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim",        default=str(DEFAULT_DIMENSION_OUTPUT), help="Dimension NetCDF file")
    p.add_argument("--spread",     default=str(DEFAULT_TIGGE_SPREAD),     help="TIGGE spread NetCDF file")
    p.add_argument("--ana",        default=str(DEFAULT_ANA),              help="Analogue spread/density NetCDF")
    p.add_argument("--k-dim",      type=int, default=50,                  help="K value for dimension (default 50)")
    p.add_argument("--estimators", nargs="+",
                   default=["d_mle", "d_mle_smooth", "d_pca", "d_pca_smooth", "d_ess", "d_ess_smooth"],
                   help="Dimension estimators to analyse")
    p.add_argument("--outdir",     default=str(DEFAULT_OUTDIR),           help="Output directory for figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    print("Loading data …")
    dim    = xr.open_dataset(args.dim)
    tigge  = xr.open_dataset(args.spread)
    ana    = xr.open_dataset(args.ana)

    spread_da = tigge["__xarray_dataarray_variable__"].sel(variable="gh")

    # ------------------------------------------------------------------ align
    common_times = np.intersect1d(
        np.intersect1d(spread_da.time.values, dim.initial_time.values),
        ana.time.values,
    )
    if len(common_times) == 0:
        raise RuntimeError("No common dates across all three datasets.")

    print(f"Common dates: {len(common_times)}")

    spread_aligned = spread_da.sel(time=common_times)   # (n, 11 steps)
    dim_aligned    = dim.sel(initial_time=common_times)
    density        = ana["density_proxy"].sel(time=common_times).values  # (n,)

    # ------------------------------------------------------------------ target
    # spread growth ratio: spread(t+h) / spread(t, h=0)
    step0 = np.timedelta64(0, "D")
    spread0 = spread_aligned.sel(step=step0).values.squeeze()  # (n,)

    y = np.column_stack([
        spread_aligned.sel(step=np.timedelta64(h, "D")).values.squeeze() / spread0
        for h in HORIZONS_DAYS
    ])  # (n, 7) — growth ratio per horizon

    # ------------------------------------------------------------------ regression
    for estimator in args.estimators:
        if estimator not in dim.data_vars:
            print(f"  [skip] {estimator} not in dimension file")
            continue

        x1 = dim_aligned[estimator].sel(K=args.k_dim).values  # (n,)
        x2 = density

        results = fit_horizon_regressions(x1, x2, y, horizons=HORIZONS_DAYS, standardize=True)

        label = f"{estimator} K={args.k_dim}"
        fig = plot_regression_coefficients(results, estimator_label=label,
                                           output=outdir / f"regression_{estimator}_K{args.k_dim}.png")

        print(f"\n{label}")
        print(f"  {'h':>3}  {'a(dim)':>8}  {'b(dens)':>8}  {'R²':>6}")
        for i, h in enumerate(HORIZONS_DAYS):
            print(f"  {h:>3}  {results['coef_a'][i]:>8.4f}  {results['coef_b'][i]:>8.4f}  {results['r2'][i]:>6.4f}")

    print(f"\nFigures saved in {outdir}/")


if __name__ == "__main__":
    main()
