"""Compute local dimensions from the raw 2-degree ERA5 z500 field."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from predictability_dimension.analogues import find_analogues
from predictability_dimension.config import DEFAULT_DIMENSION_OUTPUT, RAW_ERA5_2DEG
from predictability_dimension.dimension import dimensions_to_dataset, estimate_local_dimensions
from predictability_dimension.fields import as_numpy_matrix, field_to_state_matrix, open_geopotential_field


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_ERA5_2DEG, help="Raw ERA5 NetCDF file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIMENSION_OUTPUT, help="Output NetCDF file.")
    parser.add_argument("--variable", default=None, help="Field variable name. Inferred when omitted.")
    parser.add_argument("--level-hpa", type=float, default=500, help="Pressure level to select.")
    parser.add_argument("--step-days", type=float, default=None, help="Forecast step in days, when present.")
    parser.add_argument("--member", type=int, default=None, help="Ensemble member/number, when present.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--knn-kmax", type=int, default=500, help="Largest K for the shared neighbour search.")
    parser.add_argument("--catalog-step", type=int, default=7, help="Temporal thinning of the analogue catalogue.")
    parser.add_argument("--sample-step", type=int, default=1, help="Optional thinning of analysed times.")
    parser.add_argument("--estimators", nargs="+", default=["mle", "pca", "ess"], choices=["mle", "pca", "ess"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-anomaly", action="store_true", help="Do not remove the temporal mean at each grid point.")
    parser.add_argument("--no-standardize", action="store_true", help="Do not divide each grid point by its temporal std.")
    parser.add_argument("--no-area-weight", action="store_true", help="Do not apply sqrt(cos(latitude)) weights.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    step = None if args.step_days is None else np.timedelta64(int(args.step_days * 24), "h")

    field = open_geopotential_field(
        args.input,
        variable=args.variable,
        level_hpa=args.level_hpa,
        step=step,
        member=args.member,
    )
    states = field_to_state_matrix(
        field,
        anomaly=not args.no_anomaly,
        standardize=not args.no_standardize,
        area_weight=not args.no_area_weight,
    )
    if args.sample_step > 1:
        states = states.isel(time=slice(None, None, args.sample_step))

    matrix = as_numpy_matrix(states)
    dist, ind = find_analogues(
        matrix,
        K=args.knn_kmax,
        step_subsampling_catalogue=args.catalog_step,
        n_jobs=args.n_jobs,
    )

    dims = estimate_local_dimensions(
        matrix,
        args.k_values,
        distances=dist,
        indices=ind,
        estimators=args.estimators,
        n_jobs=args.n_jobs,
    )
    ds = dimensions_to_dataset(dims, args.k_values, states.time.data)
    ds.attrs.update(
        {
            "input_file": str(args.input),
            "field_variable": field.name,
            "source_representation": "raw gridded geopotential field",
            "catalog_step": args.catalog_step,
            "sample_step": args.sample_step,
            "anomaly": str(not args.no_anomaly),
            "standardize": str(not args.no_standardize),
            "area_weight": str(not args.no_area_weight),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
