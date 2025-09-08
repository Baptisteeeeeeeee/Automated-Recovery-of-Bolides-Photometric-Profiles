import numpy as np
import cv2
from skyfield.api import load, Topos, Star, Angle
from skyfield.data import hipparcos
from pathlib import Path
from typing import List, Tuple
import datetime
import pandas as pd


def get_stars(
    lat: float,
    lon: float,
    elevation_m: float,
    date: datetime.datetime
) -> 'pd.DataFrame':
    """
    Load stars from the Hipparcos catalog visible from a given observer location and time.

    Args:
        lat: Latitude of observer in degrees.
        lon: Longitude of observer in degrees.
        elevation_m: Elevation of observer in meters.
        date: Date and time of observation as a datetime object.

    Returns:
        A DataFrame containing stars visible above the horizon with magnitude <= 4,
        including their altitudes and azimuths in degrees.
    """
    ts = load.timescale()
    t = ts.from_datetime(date)

    BASE_DIR = Path(__file__).resolve().parents[4]
    HIP_MAIN_PATH = BASE_DIR / "utils/hip_main.dat"

    with load.open(str(HIP_MAIN_PATH)) as f:
        stars_df = hipparcos.load_dataframe(f)
    stars_df['hip'] = stars_df.index

    ra_deg = stars_df['ra_degrees'].values
    dec_deg = stars_df['dec_degrees'].values
    stars = Star(ra=Angle(degrees=ra_deg), dec=Angle(degrees=dec_deg))

    eph = load('de421.bsp')
    observer = eph['earth'] + Topos(latitude_degrees=lat, longitude_degrees=lon, elevation_m=elevation_m)

    apparent = observer.at(t).observe(stars).apparent()
    alt, az, _ = apparent.altaz()

    visible_mask = alt.degrees > 0

    stars_df = stars_df[visible_mask].copy()
    stars_df['altitude'] = alt.degrees[visible_mask]
    stars_df['azimuth'] = az.degrees[visible_mask]

    magnitude_mask = stars_df['magnitude'] <= 4
    stars_df = stars_df[magnitude_mask].reset_index(drop=True)

    return stars_df


def project_stars_on_image(
    stars_df: 'pd.DataFrame',
    image_width: int,
    image_height: int,
    az_center: float,
    alt_center: float,
    fov_h_deg: float,
    fov_v_deg: float
) -> 'pd.DataFrame':
    """
    Project star positions (altitude/azimuth) onto a 2D images plane.

    Args:
        stars_df: DataFrame containing star altitudes and azimuths in degrees.
        image_width: Width of the target images in pixels.
        image_height: Height of the target images in pixels.
        az_center: Azimuth center of the images (degrees).
        alt_center: Altitude center of the images (degrees).
        fov_h_deg: Horizontal field of view (degrees).
        fov_v_deg: Vertical field of view (degrees).

    Returns:
        A DataFrame with stars filtered to those inside the FOV, with added 'x' and 'y' pixel columns.
    """
    az_rad = np.deg2rad(stars_df['azimuth'].values)
    alt_rad = np.deg2rad(stars_df['altitude'].values)

    az_c = np.deg2rad(az_center)
    alt_c = np.deg2rad(alt_center)

    delta_az = az_rad - az_c
    delta_alt = alt_rad - alt_c

    # Handle azimuth wrap-around from 0-360° to -π to π
    delta_az = (delta_az + np.pi) % (2 * np.pi) - np.pi

    # Filter stars inside the horizontal and vertical FOV
    mask = (np.abs(delta_az) <= np.deg2rad(fov_h_deg / 2)) & (np.abs(delta_alt) <= np.deg2rad(fov_v_deg / 2))
    stars_df = stars_df[mask].copy()

    # Linear projection of coordinates onto images pixels
    x = (delta_az[mask] + np.deg2rad(fov_h_deg) / 2) * (image_width / np.deg2rad(fov_h_deg))
    y = (np.deg2rad(fov_v_deg) / 2 - delta_alt[mask]) * (image_height / np.deg2rad(fov_v_deg))

    stars_df['x'] = x
    stars_df['y'] = y

    return stars_df


def draw_stars_only(
    image_shape: Tuple[int, int],
    stars_df: 'pd.DataFrame',
) -> np.ndarray:
    """
    Render stars as white circles on a black background images.

    Args:
        image_shape: Tuple of (height, width) defining the output images size.
        stars_df: DataFrame containing 'x', 'y' pixel coordinates and 'magnitude' of stars.
        output_path: Optional path to save the generated images (not currently used).

    Returns:
        The generated RGB images as a NumPy ndarray.
    """
    image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    # Filter stars within images bounds
    valid_stars = stars_df[
        (stars_df['x'] >= 0) & (stars_df['x'] < image_shape[1]) &
        (stars_df['y'] >= 0) & (stars_df['y'] < image_shape[0])
    ].copy()

    if valid_stars.empty:
        return image

    mag_min, mag_max = valid_stars['magnitude'].min(), valid_stars['magnitude'].max()
    # Invert magnitude so smaller mag means brighter
    valid_stars['brightness'] = ((mag_max - valid_stars['magnitude']) / (mag_max - mag_min) * 255).astype(int)

    for _, star in valid_stars.iterrows():
        x, y = int(star['x']), int(star['y'])
        radius = 5
        brightness = star['brightness']
        # Color is white with brightness, here simplified as full white
        color = (brightness, brightness, brightness)
        cv2.circle(image, (x, y), radius, (255, 255, 255), -1)

    return image


def detect_clusters(
    gray_img: np.ndarray,
    threshold: int = 0,
    min_area: int = 1
) -> List[Tuple[int, int]]:
    """
    Detect bright clusters (stars) in a grayscale images using contour detection.

    Args:
        gray_img: Grayscale images as a 2D numpy array.
        threshold: Pixel intensity threshold for binarization.
        min_area: Minimum contour area to consider a cluster.

    Returns:
        List of centroid coordinates (x, y) for detected clusters.
    """
    _, binary = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clusters = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                clusters.append((cx, cy))
    return clusters
