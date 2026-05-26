"""Scatter TIGGE spread against analogue ensemble spread for h = 1 … 7 days.

Run compute_analogue_spread.py first to produce the input file.

Usage
-----
    python scripts/scatter_tigge_vs_analogue_spread.py
    python scripts/scatter_tigge_vs_analogue_spread.py \\
        --spread data/TIGGEspread.nc \\
        --ana    data/analogue_spread_density.nc \\
        --output outputs/figures/tigge_vs_ana_spread.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from predictability_dimension.config import DATA_DIR, DEFAULT_TIGGE_SPREAD
from predictability_dimension.plotting import plot_tigge_vs_analogue_spread

DEFAULT_ANA    = DATA_DIR / "analogue_spread_density.nc"
DEFAULT_OUTPUT = Path("outputs/figures/tigge_vs_analogue_spread.png")
HORIZONS_DAYS  = list(range(1, 8))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spread", default=str(DEFAULT_TIGGE_SPREAD), help="TIGGE spread NetCDF file")
    p.add_argument("--ana",    default=str(DEFAULT_ANA),          help="Analogue spread/density NetCDF file")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),       help="Output figure path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    tigge = xr.open_dataset(args.spread)
    spread_da = tigge["__xarray_dataarray_variable__"].sel(variable="gh")

    ana = xr.open_dataset(args.ana)
    ana_spread = ana["analogue_spread"]

    # Align on common times
    common_times = np.intersect1d(spread_da.time.values, ana_spread.time.values)
    if len(common_times) == 0:
        raise RuntimeError("No common dates between TIGGE and analogue spread files.")

    tigge_aligned = spread_da.sel(time=common_times)
    ana_np        = ana_spread.sel(time=common_times).values  # (n, 7)

    # horizons present in the analogue spread file
    ana_horizons_ns = ana["analogue_spread"].step.values
    ana_horizons_days = [int(h / np.timedelta64(1, "D")) for h in ana_horizons_ns]
    horizons_to_plot = [h for h in HORIZONS_DAYS if h in ana_horizons_days]
    h_indices = [ana_horizons_days.index(h) for h in horizons_to_plot]

    fig = plot_tigge_vs_analogue_spread(
        tigge_aligned,
        ana_np[:, h_indices],
        horizons_days=horizons_to_plot,
        output=output_path,
    )
    print(f"Saved → {output_path}")
    return fig


if __name__ == "__main__":
    main()
