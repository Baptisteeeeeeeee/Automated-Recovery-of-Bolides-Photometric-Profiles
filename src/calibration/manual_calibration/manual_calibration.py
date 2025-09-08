import numpy as np
import pandas as pd
from typing import Any
from .calibration.interactive_calibration import interactive_star_matching
from .calibration.matching import match_clusters_to_catalog
from .calibration.stars_utils import draw_stars_only, detect_clusters, get_stars, project_stars_on_image


def manual_calibration(
    image_path: str,
    lat: float,
    lon: float,
    elevation_m: float,
    date: Any,
    fov_h_deg: float,
    fov_v_deg: float,
    az_center: float,
    alt_center: float,
    full_mask: np.ndarray
) -> pd.DataFrame:
    """
    Perform manual calibration by matching detected stars in an images with catalog stars.

    Args:
        image_path: Path to the original images file.
        lat: Latitude of observation location in degrees.
        lon: Longitude of observation location in degrees.
        elevation_m: Elevation of observation location in meters.
        date: Date and time of the observation (datetime or compatible).
        fov_h_deg: Horizontal field of view in degrees.
        fov_v_deg: Vertical field of view in degrees.
        az_center: Azimuth center of the camera pointing in degrees.
        alt_center: Altitude center of the camera pointing in degrees.
        full_mask: Binary mask images (numpy array) with detected stars/clusters.

    Returns:
        DataFrame containing matched star clusters and catalog stars.
    """
    image_height, image_width = full_mask.shape
    stars_df = get_stars(lat, lon, elevation_m, date)
    stars_df = project_stars_on_image(
        stars_df,
        image_width,
        image_height,
        az_center,
        alt_center,
        fov_h_deg,
        fov_v_deg
    )

    image_projected = draw_stars_only((image_height, image_width), stars_df)

    clusters = detect_clusters(full_mask, threshold=0, min_area=1)

    stars_detected = [(x, y) for (x, y) in clusters]

    catalog_df = interactive_star_matching(image_path, image_projected, stars_df, stars_detected=stars_detected)

    matches = match_clusters_to_catalog(clusters, catalog_df, max_dist=15)


    return matches
