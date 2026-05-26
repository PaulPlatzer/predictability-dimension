"""Compatibility wrapper for older notebooks.

New code should import from :mod:`predictability_dimension.analogues`.
"""

from predictability_dimension.analogues import (
    compute_CRPSana,
    compute_CRPSclim,
    compute_CRPSclim_fast,
    compute_diffs,
    compute_mae_mad,
    compute_mae_monte_carlo,
    compute_theta,
    compute_velocity,
    find_analogues,
    loo_procedure,
    separate_trajectories,
)
