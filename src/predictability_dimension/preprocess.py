"""Small preprocessing helpers kept for the PCA-based legacy notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def preprocess_data(data_folder: str | Path, subsampling: int, max_horizon: int):
    """Load pseudo-PCs, normalize them, and choose target initial times."""

    data_folder = Path(data_folder)
    allpcs = xr.open_dataset(data_folder / "pcs.nc")["pseudo_pcs"]

    # Keep the relative EOF variances while setting the first PC scale to one.
    pcs_norm = allpcs / allpcs.sel(mode=0).std(dim="time")

    data_per_day = 2
    max_dh = data_per_day * max_horizon
    horizons = np.arange(np.timedelta64(0, "D"), np.timedelta64(max_horizon, "D"))
    ind_tar = np.arange(len(pcs_norm) - max_dh)[1 :: data_per_day * subsampling]

    pcs_norm.attrs["data_per_day"] = data_per_day
    return pcs_norm, ind_tar, horizons

