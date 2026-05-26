"""Compatibility wrappers for older notebooks.

New code should import from :mod:`predictability_dimension.scores`.
"""

from predictability_dimension.scores import (
    comp_CRPS_ana_horizons,
    comp_CRPS_clim,
    comp_CRPS_clim_horizons,
    comp_mad,
    comp_mae,
    compute_CRPSclim_fast,
)


def comp_CRPS_clim_L63(y, targets, window_clim, n_ref=1000, n_samples_pair=1000):
    """Legacy helper: climatological CRPS without a seasonal window."""

    return comp_CRPS_clim(y, targets, window_clim=float("inf"), n_ref=n_ref, n_samples_pairs=n_samples_pair)
