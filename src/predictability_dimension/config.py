"""Default paths used by the exploratory workflows."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

RAW_ERA5_2DEG = DATA_DIR / "ERA5_z500_1979_2023_subset_2deg.nc"
DEFAULT_DIMENSION_OUTPUT = DATA_DIR / "dim_raw_era5_2deg.nc"
DEFAULT_TIGGE_SPREAD = DATA_DIR / "TIGGEspread.nc"

