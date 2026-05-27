"""Scatter TIGGE spread growth vs analogue CRPS skill for h = 1 … 7 days.

Run compute_analogue_spread.py with --crps first to produce the CRPS data.

x-axis: TIGGE spread growth = spread(t+h) / spread(t, h=0)
y-axis: CRPS_clim(t,h) - CRPS_ana(t,h)   (positive means analogues beat clim.)

Usage
-----
    python scripts/scatter_tigge_growth_vs_crps_skill.py
    python scripts/scatter_tigge_growth_vs_crps_skill.py \\
        --spread data/TIGGEspread.nc \\
        --ana    data/analogue_spread_density.nc \\
        --output outputs/figures/tigge_growth_vs_crps_skill.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from predictability_dimension.config import DATA_DIR, DEFAULT_TIGGE_SPREAD
from predictability_dimension.plotting import plot_tigge_growth_vs_crps_skill

DEFAULT_ANA    = DATA_DIR / "analogue_spread_density.nc"
DEFAULT_OUTPUT = Path("outputs/figures/tigge_growth_vs_crps_skill.png")
HORIZONS_DAYS  = list(range(1, 8))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spread", default=str(DEFAULT_TIGGE_SPREAD))
    p.add_argument("--ana",    default=str(DEFAULT_ANA))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    print("Loading TIGGE spread …")
    tigge     = xr.open_dataset(args.spread)
    spread_da = tigge["__xarray_dataarray_variable__"].sel(variable="gh")

    print("Loading analogue CRPS data …")
    ana = xr.open_dataset(args.ana)

    for var in ("crps_ana", "crps_clim"):
        if var not in ana:
            raise RuntimeError(
                f"Variable '{var}' not found in {args.ana}.\n"
                "Re-run compute_analogue_spread.py with the --crps flag."
            )

    # ── align on common times ─────────────────────────────────────────────────
    common_times = np.intersect1d(spread_da.time.values, ana.time.values)
    if len(common_times) == 0:
        raise RuntimeError("No common dates between TIGGE and analogue files.")

    spread_aligned = spread_da.sel(time=common_times)
    crps_ana  = ana["crps_ana"].sel(time=common_times)
    crps_clim = ana["crps_clim"].sel(time=common_times)

    # Pick horizons present in both files
    ana_steps_days = [int(s / np.timedelta64(1, "D")) for s in crps_ana.step.values]
    horizons = [h for h in HORIZONS_DAYS if h in ana_steps_days]
    h_idx    = [ana_steps_days.index(h) for h in horizons]

    crps_ana_np  = crps_ana.values[:, h_idx]   # (n, len(horizons))
    crps_clim_np = crps_clim.values[:, h_idx]

    # ── plot ──────────────────────────────────────────────────────────────────
    fig = plot_tigge_growth_vs_crps_skill(
        spread_aligned,
        crps_ana_np,
        crps_clim_np,
        horizons_days=horizons,
        output=output_path,
    )
    print(f"Saved → {output_path}")
    return fig


if __name__ == "__main__":
    main()
