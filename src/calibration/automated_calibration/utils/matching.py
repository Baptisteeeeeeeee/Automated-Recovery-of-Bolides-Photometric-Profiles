from pathlib import Path
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from skyfield.api import load
from skyfield.data import hipparcos
from typing import List, Tuple


def match_coords(image: np.ndarray, stars: List[Tuple[float, float]], wcs: WCS) -> pd.DataFrame:
    """
    Matches detected star pixel positions to catalog stars using WCS coordinates.

    Parameters
    ----------
    image : np.ndarray
        Original images (used for width and height limits).

    stars : list of (float, float)
        Detected star positions in pixel space.

    wcs : WCS
        Astrometric WCS object for coordinate conversion.

    Returns
    -------
    pd.DataFrame
        DataFrame with matched catalog star info and pixel detection positions.
    """
    h, w = image.shape[:2]
    stars_df = get_stars()
    stars_df = project_stars_with_wcs(stars_df, wcs)

    # Keep only stars that fall within images bounds
    mask = ((stars_df['x'] > 0) & (stars_df['x'] < w)) & ((stars_df['y'] > 0) & (stars_df['y'] < h))
    stars_df_filtered = stars_df[mask].copy()

    # Convert detected stars to SkyCoord
    pixel_coords = np.array(stars)
    sky_coords_detected = SkyCoord.from_pixel(pixel_coords[:, 0], pixel_coords[:, 1], wcs=wcs)

    # Convert catalog stars to SkyCoord
    stars_catalog = SkyCoord(
        ra=stars_df_filtered['ra_degrees'].values * u.deg,
        dec=stars_df_filtered['dec_degrees'].values * u.deg
    )

    # Match each detected star to nearest catalog star within threshold
    idx, sep2d, _ = sky_coords_detected.match_to_catalog_sky(stars_catalog)
    max_sep = 1 * u.deg
    matches = [(i, j, sep) for i, (j, sep) in enumerate(zip(idx, sep2d)) if sep < max_sep]
    print(f"[INFO] {len(matches)} valid matches found (threshold = {max_sep}).")

    # Create matched DataFrame
    matched_rows = []
    for i_detected, j_catalog, sep in matches:
        x_det, y_det = pixel_coords[i_detected]
        matched_star = stars_df_filtered.iloc[j_catalog].copy()

        matched_star['x_detected'] = x_det
        matched_star['y_detected'] = y_det
        matched_star['separation_deg'] = sep.deg

        matched_rows.append(matched_star)

    return pd.DataFrame(matched_rows).reset_index(drop=True)


def get_stars() -> pd.DataFrame:
    """
    Loads HIPPARCOS star catalog and filters for stars with magnitude <= 4.

    Returns
    -------
    pd.DataFrame
        DataFrame of bright stars with HIP identifiers and coordinates.
    """
    BASE_DIR = Path(__file__).resolve().parents[4]
    HIP_MAIN_PATH = BASE_DIR / "utils/hip_main.dat"

    with load.open(str(HIP_MAIN_PATH)) as f:
        stars_df = hipparcos.load_dataframe(f)

    stars_df['hip'] = stars_df.index

    # Keep only bright stars (optional, improves matching quality)
    stars_df = stars_df[stars_df['magnitude'] <= 4].reset_index(drop=True)

    return stars_df


def project_stars_with_wcs(stars_df: pd.DataFrame, wcs: WCS) -> pd.DataFrame:
    """
    Projects RA/DEC star coordinates into pixel space using WCS.

    Parameters
    ----------
    stars_df : pd.DataFrame
        DataFrame with 'ra_degrees' and 'dec_degrees' columns.

    wcs : WCS
        World Coordinate System object.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with 'x', 'y' columns (images pixel coordinates).
    """
    coords = SkyCoord(
        ra=stars_df['ra_degrees'].values * u.deg,
        dec=stars_df['dec_degrees'].values * u.deg
    )

    try:
        x, y = wcs.world_to_pixel(coords)
    except Exception as e:
        print(f"[ERROR] WCS projection failed: {e}")
        return pd.DataFrame()

    stars_df['x'] = x
    stars_df['y'] = y

    # Remove invalid coordinates
    stars_df = stars_df.replace([np.inf, -np.inf], np.nan)
    stars_df = stars_df.dropna(subset=['x', 'y'])

    return stars_df
