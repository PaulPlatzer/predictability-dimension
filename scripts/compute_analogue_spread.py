"""Compute analogue ensemble spread, multi-K density proxy, CRPS and neighbourhood skill.

For each date t in the TIGGE forecast archive, find K nearest analogues in
the ERA5 catalogue and then compute:

  * density_proxy(t, K_d)        = mean distance to K_d analogues
  * analogue_spread(t, h)        = RMS spread of evolved analogue states at t+h
  * crps_ana(t, h)               = analogue CRPS vs verified state  [with --crps]
  * crps_clim(t, h)              = climatological CRPS              [with --crps]
  * neighbourhood_skill(t, h)    = mean over K analogues of         [with --crps-catalogue]
                                   [CRPS_clim(t_k,h) - CRPS_ana(t_k,h)]

All outputs are written to a single NetCDF file for downstream analysis.

Usage
-----
    # Minimal (spread + density only)
    python scripts/compute_analogue_spread.py

    # Full: add CRPS for target dates
    python scripts/compute_analogue_spread.py --crps

    # Full: add CRPS for target AND catalogue dates → neighbourhood_skill
    python scripts/compute_analogue_spread.py --crps --crps-catalogue --n-jobs 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from predictability_dimension.analogues import (
    compute_analogue_spread_horizons,
    compute_crps_ana_horizons,
    compute_crps_clim_horizons,
    compute_density_proxy,
    find_analogues,
)
from predictability_dimension.config import DATA_DIR, DEFAULT_TIGGE_SPREAD, RAW_ERA5_2DEG
from predictability_dimension.fields import (
    as_numpy_matrix,
    field_to_state_matrix,
    open_geopotential_field,
)

SPREAD_HORIZONS = list(range(0, 8))   # h=0 needed for growth normalisation
CRPS_HORIZONS   = list(range(1, 8))   # CRPS only for h≥1 (comparing future states)
K_DENSITY_LIST  = [10, 20, 50, 100]
DEFAULT_OUTPUT = DATA_DIR / "analogue_spread_density.nc"
CATALOG_STEP   = 7
DT_LOO         = 15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input",            default=str(RAW_ERA5_2DEG))
    p.add_argument("--spread",           default=str(DEFAULT_TIGGE_SPREAD))
    p.add_argument("--output",           default=str(DEFAULT_OUTPUT))
    p.add_argument("--kmax",             type=int, default=100)
    p.add_argument("--n-jobs",           type=int, default=1)
    p.add_argument("--crps",             action="store_true",
                   help="Compute CRPS_ana and CRPS_clim for TIGGE target dates")
    p.add_argument("--crps-catalogue",   action="store_true",
                   help="Also compute CRPS for catalogue dates → neighbourhood_skill "
                        "(implies --crps)")
    p.add_argument("--n-ref",            type=int, default=300,
                   help="Climatological sample size for CRPS_clim")
    return p.parse_args()


# ─────────────────────────────────── helpers ──────────────────────────────────

def _crps_for_dates(
    X: np.ndarray,
    tar_idx: np.ndarray,
    cat_idx: np.ndarray,
    k: int,
    n_ref: int,
    n_jobs: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Find analogues for tar_idx, then compute CRPS_ana and CRPS_clim."""
    print(f"  Searching analogues for {len(tar_idx)} {label} dates …")
    dist, ind = find_analogues(
        X, ind_cat=cat_idx, ind_tar=tar_idx,
        K=k, loo=True, dt_loo=DT_LOO, n_jobs=n_jobs,
    )
    print(f"  Computing CRPS_ana ({label}) …")
    crps_ana  = compute_crps_ana_horizons(X, tar_idx, ind, CRPS_HORIZONS, k=k)
    print(f"  Computing CRPS_clim ({label}, n_ref={n_ref}) …")
    crps_clim = compute_crps_clim_horizons(X, tar_idx, CRPS_HORIZONS, n_ref=n_ref)
    return crps_ana, crps_clim


def _neighbourhood_skill(
    ind_tigge: np.ndarray,     # (n_tigge, K)
    cat_idx: np.ndarray,       # (n_cat,)
    crps_skill_cat: np.ndarray,  # (n_cat, 7)
    k: int,
) -> np.ndarray:
    """Average CRPS skill of K analogues over catalogue dates."""
    n_times_full = int(cat_idx.max()) + 1   # size of lookup array

    # Build fast lookup: absolute ERA5 index → catalogue position
    cat_pos_lookup = np.full(n_times_full, -1, dtype=np.intp)
    for j, idx in enumerate(cat_idx):
        cat_pos_lookup[idx] = j

    ind_k = ind_tigge[:, :k].astype(np.intp)  # (n_tigge, k)

    # All indices should be in the catalogue by construction
    cat_positions = cat_pos_lookup[ind_k]      # (n_tigge, k)

    # Look up skill for each (target, analogue) pair: (n_tigge, k, 7)
    neighbour_skills = crps_skill_cat[cat_positions]

    # Average over k, ignoring NaN
    return np.nanmean(neighbour_skills, axis=1)  # (n_tigge, 7)


# ─────────────────────────────────── main ─────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.crps_catalogue:
        args.crps = True  # catalogue CRPS implies target CRPS

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── ERA5 field ────────────────────────────────────────────────────────────
    print("Loading ERA5 field …")
    field      = open_geopotential_field(args.input)
    states     = field_to_state_matrix(field, anomaly=True, standardize=False, area_weight=True)
    X          = as_numpy_matrix(states)
    era5_times = states.time.values

    # ── TIGGE dates → target indices ─────────────────────────────────────────
    print("Loading TIGGE spread …")
    tigge       = xr.open_dataset(args.spread)
    tigge_times = tigge.time.values

    era5_index        = {t: i for i, t in enumerate(era5_times)}
    tar_idx           = np.array([era5_index[t] for t in tigge_times if t in era5_index])
    valid_tigge_times = np.array([t for t in tigge_times if t in era5_index])

    if len(tar_idx) == 0:
        raise RuntimeError("No common dates between TIGGE and ERA5.")

    kmax    = max(args.kmax, max(K_DENSITY_LIST))
    cat_idx = np.arange(0, len(era5_times), CATALOG_STEP)
    k_use   = min(50, kmax)

    # ── Analogues for TIGGE targets ───────────────────────────────────────────
    print(f"Searching K={kmax} analogues for {len(tar_idx)} TIGGE targets …")
    dist, ind = find_analogues(
        X, ind_cat=cat_idx, ind_tar=tar_idx,
        K=kmax, loo=True, dt_loo=DT_LOO, n_jobs=args.n_jobs,
    )

    # ── Density proxy (multiple K) ────────────────────────────────────────────
    print("Computing density proxy …")
    k_density_vals = [k for k in K_DENSITY_LIST if k <= dist.shape[1]]
    density_matrix = np.column_stack(
        [compute_density_proxy(dist, k=k) for k in k_density_vals]
    ).astype("float32")

    # ── Analogue spread (including h=0 for growth normalisation) ─────────────
    print(f"Computing analogue spread (K={k_use}, h={SPREAD_HORIZONS}) …")
    ana_spread = compute_analogue_spread_horizons(X, ind, SPREAD_HORIZONS, k=k_use)

    # ── Build dataset ─────────────────────────────────────────────────────────
    steps = np.array([np.timedelta64(h, "D") for h in SPREAD_HORIZONS])

    ds = xr.Dataset(
        {
            "analogue_spread": xr.DataArray(
                ana_spread.astype("float32"),
                dims=["time", "step"],
                coords={"time": valid_tigge_times, "step": steps},
                attrs={"long_name": "RMS analogue ensemble spread (state-vector space)"},
            ),
            "density_proxy": xr.DataArray(
                density_matrix,
                dims=["time", "K_density"],
                coords={"time": valid_tigge_times, "K_density": k_density_vals},
                attrs={"long_name": "Mean distance to K nearest analogues"},
            ),
        },
        attrs={"kmax": kmax, "catalog_step": CATALOG_STEP, "dt_loo": DT_LOO,
               "era5_input": str(args.input)},
    )

    # CRPS and neighbourhood_skill use h=1..7 only
    crps_steps = np.array([np.timedelta64(h, "D") for h in CRPS_HORIZONS])

    # ── CRPS for TIGGE targets (optional) ────────────────────────────────────
    if args.crps:
        print("\nComputing CRPS for TIGGE target dates …")
        crps_ana, crps_clim = _crps_for_dates(
            X, tar_idx, cat_idx, k=k_use, n_ref=args.n_ref,
            n_jobs=args.n_jobs, label="TIGGE"
        )
        ds["crps_ana"]  = xr.DataArray(
            crps_ana.astype("float32"),  dims=["time", "step_crps"],
            coords={"time": valid_tigge_times, "step_crps": crps_steps},
            attrs={"long_name": "Analogue CRPS (state-vector space)"},
        )
        ds["crps_clim"] = xr.DataArray(
            crps_clim.astype("float32"), dims=["time", "step_crps"],
            coords={"time": valid_tigge_times, "step_crps": crps_steps},
            attrs={"long_name": "Climatological CRPS (state-vector space)"},
        )

    # ── CRPS for catalogue dates → neighbourhood skill (optional) ─────────────
    if args.crps_catalogue:
        print("\nComputing CRPS for catalogue dates …")
        crps_ana_cat, crps_clim_cat = _crps_for_dates(
            X, cat_idx, cat_idx, k=k_use, n_ref=args.n_ref,
            n_jobs=args.n_jobs, label="catalogue"
        )
        crps_skill_cat = crps_clim_cat - crps_ana_cat  # (n_cat, 7)

        print("Computing neighbourhood skill …")
        neigh_skill = _neighbourhood_skill(ind, cat_idx, crps_skill_cat, k=k_use)

        ds["neighbourhood_skill"] = xr.DataArray(
            neigh_skill.astype("float32"), dims=["time", "step_crps"],
            coords={"time": valid_tigge_times, "step_crps": crps_steps},
            attrs={"long_name": "Mean CRPS skill of K analogues (neighbourhood predictability)"},
        )

    ds.to_netcdf(output_path)
    print(f"\nSaved → {output_path}")
    print(f"Variables: {list(ds.data_vars)}")


if __name__ == "__main__":
    main()
