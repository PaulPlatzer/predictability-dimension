"""Compute analogue ensemble spread and density proxy for TIGGE dates.

For each date t in the TIGGE forecast archive, find K=50 nearest analogues
in the ERA5 catalogue and then:

  * density proxy(t)      = mean distance to the K analogues at time t
  * analogue_spread(t, h) = RMS spread of the K evolved analogue states
                             at time t + h  (h = 1 … 7 days)

Results are written to a single NetCDF file that can be loaded alongside
the TIGGE spread and dimension estimates for regression analysis.

Usage
-----
    python scripts/compute_analogue_spread.py
    python scripts/compute_analogue_spread.py --k 50 --kmax 100 --n-jobs 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from predictability_dimension.analogues import (
    compute_analogue_spread_horizons,
    compute_density_proxy,
    find_analogues,
)
from predictability_dimension.config import DATA_DIR, DEFAULT_TIGGE_SPREAD, RAW_ERA5_2DEG
from predictability_dimension.fields import (
    as_numpy_matrix,
    field_to_state_matrix,
    open_geopotential_field,
)

HORIZONS_DAYS = list(range(1, 8))   # 1 … 7 days
DEFAULT_OUTPUT = DATA_DIR / "analogue_spread_density.nc"
CATALOG_STEP   = 7   # keep consistent with compute_raw_dimension.py
DT_LOO         = 15  # days excluded around the target date (LOO window)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",    default=str(RAW_ERA5_2DEG),      help="ERA5 z500 NetCDF file")
    p.add_argument("--spread",   default=str(DEFAULT_TIGGE_SPREAD), help="TIGGE spread NetCDF file")
    p.add_argument("--output",   default=str(DEFAULT_OUTPUT),     help="Output NetCDF file")
    p.add_argument("--k",        type=int, default=50,            help="Number of analogues (default 50)")
    p.add_argument("--kmax",     type=int, default=100,           help="Fetch kmax neighbours then trim to k")
    p.add_argument("--n-jobs",   type=int, default=1,             help="Parallel jobs for KNN search")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("Loading ERA5 field …")
    field  = open_geopotential_field(args.input)
    states = field_to_state_matrix(field, anomaly=True, standardize=False, area_weight=True)
    X      = as_numpy_matrix(states)                    # (16356, 1220)
    era5_times = states.time.values                     # datetime64[ns]

    print("Loading TIGGE spread …")
    tigge      = xr.open_dataset(args.spread)
    tigge_times = tigge.time.values                     # (4762,) subset of ERA5

    # ------------------------------------------------------------------ indices
    # Positions of TIGGE dates in the ERA5 time axis (targets)
    era5_time_index = {t: i for i, t in enumerate(era5_times)}
    tar_idx = np.array([era5_time_index[t] for t in tigge_times if t in era5_time_index])
    valid_tigge_times = np.array([t for t in tigge_times if t in era5_time_index])

    if len(tar_idx) == 0:
        raise RuntimeError("No common dates between TIGGE and ERA5.")

    cat_idx = np.arange(0, len(era5_times), CATALOG_STEP)  # catalogue (every 7th day)

    print(f"Searching K={args.k} analogues for {len(tar_idx)} TIGGE dates "
          f"(catalogue size: {len(cat_idx)}, LOO window: ±{DT_LOO} days) …")
    dist, ind = find_analogues(
        X,
        ind_cat=cat_idx,
        ind_tar=tar_idx,
        K=args.kmax,
        loo=True,
        dt_loo=DT_LOO,
        n_jobs=args.n_jobs,
    )
    # dist, ind: (n_targets, kmax)

    # ------------------------------------------------------------------ density
    print("Computing density proxy …")
    density = compute_density_proxy(dist, k=args.k)          # (n_targets,)

    # ------------------------------------------------------------------ spread
    print(f"Computing analogue spread at horizons {HORIZONS_DAYS} days …")
    ana_spread = compute_analogue_spread_horizons(            # (n_targets, 7)
        X, ind, horizons=HORIZONS_DAYS, k=args.k
    )

    # ------------------------------------------------------------------ save
    steps = np.array([np.timedelta64(h, "D") for h in HORIZONS_DAYS])

    ds = xr.Dataset(
        {
            "analogue_spread": xr.DataArray(
                ana_spread.astype("float32"),
                dims=["time", "step"],
                coords={"time": valid_tigge_times, "step": steps},
                attrs={"long_name": "RMS analogue ensemble spread", "units": "m^2 s^-2 (anomaly space)"},
            ),
            "density_proxy": xr.DataArray(
                density.astype("float32"),
                dims=["time"],
                coords={"time": valid_tigge_times},
                attrs={"long_name": "Mean distance to K nearest analogues (inverse density)", "units": "m^2 s^-2 (anomaly space)"},
            ),
        },
        attrs={
            "k": args.k,
            "catalog_step": CATALOG_STEP,
            "dt_loo": DT_LOO,
            "era5_input": str(args.input),
        },
    )

    ds.to_netcdf(output_path)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
