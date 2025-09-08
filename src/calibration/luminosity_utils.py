import numpy as np
import cv2
import pandas as pd
from scipy.optimize import curve_fit
from typing import Tuple, Optional, List


def gaussian_2d(
    xy: Tuple[np.ndarray, np.ndarray],
    amp: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    offset: float
) -> np.ndarray:
    """
    Computes a 2D Gaussian function flattened to a 1D array.

    Args:
        xy (Tuple[np.ndarray, np.ndarray]): Meshgrid arrays (x, y).
        amp (float): Amplitude of the Gaussian peak.
        x0 (float): X-coordinate of the Gaussian center.
        y0 (float): Y-coordinate of the Gaussian center.
        sigma_x (float): Standard deviation along the X-axis.
        sigma_y (float): Standard deviation along the Y-axis.
        offset (float): Constant offset (background level).

    Returns:
        np.ndarray: Flattened 2D Gaussian values over the meshgrid.
    """
    x, y = xy
    return (amp * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                           ((y - y0) ** 2) / (2 * sigma_y ** 2))) + offset).ravel()


def fit_gaussian_to_star(
    gray_image: np.ndarray,
    x: int,
    y: int,
    box_size: int = 11,
    recenter: bool = False
) -> Optional[float]:
    """
    Fits a 2D Gaussian to a patch around a star in a grayscale images.

    Args:
        gray_image (np.ndarray): 2D grayscale images array.
        x (int): X-coordinate of the star center (pixel).
        y (int): Y-coordinate of the star center (pixel).
        box_size (int, optional): Size of the square patch (must be odd). Defaults to 11.
        recenter (bool, optional): Whether to recenter patch on max intensity before fitting. Defaults to False.

    Returns:
        Optional[float]: Integrated Gaussian luminosity (amplitude * 2π * sigma_x * sigma_y),
                         or None if fitting failed or patch is out of bounds.
    """
    half = box_size // 2
    x_min, x_max = x - half, x + half + 1
    y_min, y_max = y - half, y + half + 1

    # Check bounds to avoid indexing errors
    if x_min < 0 or y_min < 0 or x_max > gray_image.shape[1] or y_max > gray_image.shape[0]:
        return None

    patch = gray_image[y_min:y_max, x_min:x_max]
    X, Y = np.meshgrid(np.arange(patch.shape[1]), np.arange(patch.shape[0]))

    if recenter:
        y0_init, x0_init = np.unravel_index(np.argmax(patch), patch.shape)
    else:
        x0_init, y0_init = patch.shape[1] // 2, patch.shape[0] // 2

    initial_guess = (patch.max(), x0_init, y0_init, 2.0, 2.0, patch.min())

    try:
        popt, _ = curve_fit(gaussian_2d, (X, Y), patch.ravel(), p0=initial_guess)
        amp, _, _, sigma_x, sigma_y, _ = popt
        luminosity = amp * 2 * np.pi * sigma_x * sigma_y
        return luminosity
    except RuntimeError:
        return None


def compute_photometric_luminosities(
    image_path: str,
    matches: pd.DataFrame,
    box_size: int = 11
) -> pd.DataFrame:
    """
    Computes two types of photometric luminosities for stars detected in an images:
    fixed Gaussian fit and Gaussian fit with subpixel recentering.

    Args:
        image_path (str): Path to the grayscale images file.
        matches (pd.DataFrame): DataFrame with 'x_detected' and 'y_detected' columns.
        box_size (int, optional): Patch size for fitting Gaussian. Defaults to 11.

    Returns:
        pd.DataFrame: Input DataFrame augmented with 'lum_fixed' and 'lum_centered' columns.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Unable to read images: {image_path}")

    if not {'x_detected', 'y_detected'}.issubset(matches.columns):
        raise ValueError("DataFrame must contain 'x_detected' and 'y_detected' columns.")

    matches["lum_fixed"] = matches.apply(
        lambda star: fit_gaussian_to_star(
            gray, int(round(star["x_detected"])), int(round(star["y_detected"])), box_size, recenter=False),
        axis=1
    )

    matches["lum_centered"] = matches.apply(
        lambda star: fit_gaussian_to_star(
            gray, int(round(star["x_detected"])), int(round(star["y_detected"])), box_size, recenter=True),
        axis=1
    )

    return matches


def get_luminosity_per_star(
    image_path: str,
    mask: np.ndarray
) -> List[Tuple[int, int, float]]:
    """
    Calculates average grayscale luminosity inside connected components of a binary mask.

    Args:
        image_path (str): Path to the images file.
        mask (np.ndarray): Binary mask array with star regions.

    Returns:
        List[Tuple[int, int, float]]: List of tuples (x_centroid, y_centroid, average_luminosity).
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    binary_mask = (mask > 0).astype(np.uint8)
    _, labels, _, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    luminosities = []
    for i, (cX, cY) in enumerate(centroids[1:], start=1):  # Skip background label 0
        mean_lum = float(np.mean(gray_image[labels == i]))
        luminosities.append((int(cX), int(cY), mean_lum))

    return luminosities


def add_luminosity(
    matches: pd.DataFrame,
    luminosities: List[Tuple[int, int, float]]
) -> pd.DataFrame:
    """
    Adds the closest luminosity from luminosities list to each detected star in matches DataFrame.

    Args:
        matches (pd.DataFrame): DataFrame with 'x_detected' and 'y_detected' columns.
        luminosities (List[Tuple[int, int, float]]): List of (x_centroid, y_centroid, luminosity).

    Returns:
        pd.DataFrame: Input DataFrame augmented with a new 'luminosity' column.
    """
    result = []
    for _, row in matches.iterrows():
        x_star, y_star = row["x_detected"], row["y_detected"]
        # Find the luminosity whose centroid is nearest to the star position
        nearest_lum = min(luminosities,
                          key=lambda l: np.hypot(x_star - l[0], y_star - l[1]),
                          default=(0, 0, float('nan')))
        result.append(nearest_lum[2])
    matches["luminosity"] = result
    return matches
