from .utils.astrometry_utils import run_local_astrometry_from_image, extract_wcs_from_fits
from .utils.matching import match_coords
import pandas as pd


def automated_calibration(full_mask_uint8, stars_px, image):
    try:
        fits_with_wcs = run_local_astrometry_from_image(full_mask_uint8)
        wcs, data = extract_wcs_from_fits(fits_with_wcs)
        matches = match_coords(image, stars_px, wcs)
    except Exception as e:
        matches = pd.DataFrame()

    if not matches.empty:
        print(matches)
    return matches

