"""CRPS utilities used by the exploratory notebooks."""

from __future__ import annotations

import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from .analogues import compute_CRPSana


def comp_mad(y: xr.DataArray, n_sample_pairs: int = 10000, random_seed: int = 1312) -> float:
    """Monte-Carlo estimate of E|Y - Y*|."""

    rng = np.random.default_rng(random_seed)
    n = len(y)
    if n_sample_pairs < n:
        i1 = rng.choice(n, size=n_sample_pairs, replace=False)
        i2 = rng.choice(n, size=n_sample_pairs, replace=False)
    else:
        i1 = np.arange(n)
        i2 = rng.permutation(n)
    return float(np.mean(np.abs(y.data[i1] - y.data[i2])))


def comp_mae(y: xr.DataArray, target: xr.DataArray, n_ref: int = 1000, random_seed: int = 1312) -> float:
    """Monte-Carlo estimate of E|Y - target|."""

    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(y), size=min(n_ref, len(y)), replace=False)
    return float(np.mean(np.abs(y.data[idx] - np.expand_dims(target.data, axis=0))))


def comp_CRPS_clim(
    y: xr.DataArray,
    targets: xr.DataArray,
    window_clim: float,
    n_ref: int = 1000,
    n_samples_pairs: int = 10000,
    random_seed: int = 1312,
) -> np.ndarray:
    """CRPS of targets against a seasonal climatology."""

    crps = np.full(len(targets), np.nan)
    absdiff = np.abs(y.time.dt.dayofyear - targets.target_time.dt.dayofyear)
    within_window = np.logical_or(absdiff < window_clim, absdiff > 366 - window_clim)

    for i, target in enumerate(tqdm(targets, desc="CRPS climatology")):
        restricted_y = y.where(within_window.isel(target_time=i), drop=True)
        mad = comp_mad(restricted_y, n_samples_pairs, random_seed=random_seed + i)
        mae = comp_mae(restricted_y, target, n_ref, random_seed=random_seed + i)
        crps[i] = mae - 0.5 * mad

    return crps


def comp_CRPS_clim_horizons(
    pcs_norm: xr.DataArray,
    ind_tar: np.ndarray,
    horizons: np.ndarray,
    window_clim: float,
) -> xr.DataArray:
    """Compute climatological CRPS for each forecast horizon."""

    crps = xr.DataArray(
        np.full((len(horizons), len(ind_tar)), np.nan),
        dims=("horizon", "initial_time"),
        coords={"horizon": horizons, "initial_time": pcs_norm.time.isel(time=ind_tar).data},
        attrs={"window_clim": window_clim},
    )

    data_per_day = int(pcs_norm.attrs["data_per_day"])
    max_dh = data_per_day * int(horizons.max() // np.timedelta64(1, "D"))
    n_time = len(pcs_norm)

    for horizon in horizons:
        dh = data_per_day * int(horizon // np.timedelta64(1, "D"))
        dataset_y = pcs_norm[dh : n_time - (max_dh - dh)]
        targets = dataset_y[ind_tar].rename({"time": "target_time"})
        crps.loc[{"horizon": horizon}] = comp_CRPS_clim(dataset_y, targets, window_clim)

    return crps


def comp_CRPS_ana_horizons(
    pcs_norm: xr.DataArray,
    ind_tar: np.ndarray,
    horizons: np.ndarray,
    Kvalues: list[int],
    n_jobs: int = 1,
) -> xr.DataArray:
    """Compute analogue-forecast CRPS for each horizon and K."""

    crps = xr.DataArray(
        np.full((len(horizons), len(Kvalues), len(ind_tar)), np.nan),
        dims=("horizon", "K", "initial_time"),
        coords={"horizon": horizons, "K": Kvalues, "initial_time": pcs_norm.time[ind_tar].data},
    )

    data_per_day = int(pcs_norm.attrs["data_per_day"])
    max_dh = data_per_day * int(horizons.max() // np.timedelta64(1, "D"))
    n_time = len(pcs_norm)

    for horizon in horizons:
        dh = data_per_day * int(horizon // np.timedelta64(1, "D"))
        dataset_x = pcs_norm[:-max_dh]
        dataset_y = pcs_norm[dh : n_time - (max_dh - dh)]
        catalogue_step = data_per_day * 4
        for K in tqdm(Kvalues, desc=f"CRPS analogue h={horizon}"):
            crps.loc[{"horizon": horizon, "K": K}] = compute_CRPSana(
                dataset_x.data,
                dataset_y.data,
                ind_tar=ind_tar,
                loo=True,
                dt_loo=data_per_day * 30,
                K=K,
                step_subsampling_catalogue=catalogue_step,
                n_jobs=n_jobs,
            )

    return crps


def compute_CRPSclim_fast(
    dataset_y: np.ndarray,
    ind_tar: np.ndarray | None = None,
    n_sample_pairs: int = 10000,
    n_ref: int = 1000,
    random_seed: int = 1312,
) -> np.ndarray:
    """Fast climatological CRPS approximation for plain NumPy arrays."""

    rng = np.random.default_rng(random_seed)
    dataset_y = np.asarray(dataset_y)
    n = len(dataset_y)
    if ind_tar is None:
        ind_tar = np.arange(n)

    i1 = rng.integers(0, n, size=n_sample_pairs)
    i2 = rng.integers(0, n, size=n_sample_pairs)
    mad = np.mean(np.abs(dataset_y[i1] - dataset_y[i2]))

    ref = rng.choice(n, size=min(n_ref, n), replace=False)
    mae = np.mean(np.abs(dataset_y[ind_tar][:, None, :] - dataset_y[ref][None, :, :]), axis=(1, 2))
    return mae - 0.5 * mad

