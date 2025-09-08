import numpy as np
from typing import List, Tuple, Optional, Union
from scipy.optimize import curve_fit
from skimage.measure import regionprops
import numpy as np

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
    2D Gaussian function.

    Args:
        xy (Tuple[np.ndarray, np.ndarray]): Meshgrid arrays (X, Y).
        amp (float): Amplitude of the Gaussian.
        x0 (float): X-center of the Gaussian.
        y0 (float): Y-center of the Gaussian.
        sigma_x (float): Standard deviation along the X axis.
        sigma_y (float): Standard deviation along the Y axis.
        offset (float): Constant offset.

    Returns:
        np.ndarray: Flattened 2D Gaussian evaluated at xy.
    """
    x, y = xy
    return (amp * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                           ((y - y0) ** 2) / (2 * sigma_y ** 2))) + offset).ravel()


def fit_gaussian_to_source(
    gray_image: np.ndarray,
    x: int,
    y: int,
    box_size: int = 21,
    recenter: bool = True
) -> float:
    """
    Fits a 2D Gaussian to a source in a grayscale images patch.

    Args:
        gray_image (np.ndarray): Grayscale images.
        x (int): X-coordinate of the source center.
        y (int): Y-coordinate of the source center.
        box_size (int): Size of the square box to extract the patch.
        recenter (bool): Whether to recenter on the local maximum before fitting.

    Returns:
        float: Integrated flux (amplitude * 2π * σ_x * σ_y) or NaN on failure.
    """
    half = box_size // 2
    x_min, x_max = x - half, x + half + 1
    y_min, y_max = y - half, y + half + 1

    if x_min < 0 or y_min < 0 or x_max > gray_image.shape[1] or y_max > gray_image.shape[0]:
        return float("nan")

    patch = gray_image[y_min:y_max, x_min:x_max]
    X, Y = np.meshgrid(np.arange(patch.shape[1]), np.arange(patch.shape[0]))

    if recenter:
        max_y, max_x = np.unravel_index(np.argmax(patch), patch.shape)
        x0_init, y0_init = max_x, max_y
    else:
        x0_init, y0_init = patch.shape[1] // 2, patch.shape[0] // 2

    initial_guess = (
        patch.max(), x0_init, y0_init,
        2.0, 2.0,
        patch.min()
    )

    try:
        popt, _ = curve_fit(gaussian_2d, (X, Y), patch.ravel(), p0=initial_guess)
        amp, _, _, sigma_x, sigma_y, _ = popt
        return amp * 2 * np.pi * sigma_x * sigma_y
    except RuntimeError:
        return float("nan")


def fit_multiple_gaussians_along_mask(
    gray_image: np.ndarray,
    mask: np.ndarray,
    box_size: int = 21,
    step: int = 10
) -> Tuple[float, List[Tuple[int, int, int]], Optional[Tuple[int, int]]]:
    """
    Fits multiple 2D Gaussians along the main axis of a binary mask.

    Args:
        gray_image (np.ndarray): Grayscale input images.
        mask (np.ndarray): Binary mask defining the region of interest.
        box_size (int): Size of the window for Gaussian fitting.
        step (int): Spacing between fit locations along the mask axis.

    Returns:
        Tuple containing:
            - total_luminosity (float): Sum of all fitted Gaussian fluxes.
            - fit_zones (List[Tuple[int, int, int]]): List of bounding boxes for fits (x, y, size).
            - extreme_point (Optional[Tuple[int, int]]): Coordinates of the last fitted position.
    """
    PADDING = box_size
    gray_padded = np.pad(gray_image, PADDING, mode='reflect')
    mask_padded = np.pad(mask, PADDING, mode='constant', constant_values=0)

    props = regionprops(mask_padded.astype(np.uint8))
    if len(props) == 0:
        return float("nan"), [], None

    prop = props[0]
    coords = np.array(prop.coords)
    coords_mean = coords.mean(axis=0)
    centered_coords = coords - coords_mean
    _, _, Vt = np.linalg.svd(centered_coords)
    direction = Vt[0]

    min_proj, max_proj = (centered_coords @ direction).min(), (centered_coords @ direction).max()
    positions = np.arange(min_proj, max_proj, step)

    luminosity_sum = 0.0
    fit_zones: List[Tuple[int, int, int]] = []
    extreme_point: Optional[Tuple[int, int]] = None

    for pos in positions:
        point = coords_mean + pos * direction
        y, x = int(point[0]), int(point[1])
        lum = fit_gaussian_to_source(gray_padded, x, y, box_size=box_size, recenter=True)
        if not np.isnan(lum):
            luminosity_sum += lum
            half = box_size // 2
            fit_zones.append((x - half - PADDING, y - half - PADDING, box_size))
            extreme_point = (x - PADDING, y - PADDING)

    if len(fit_zones) == 0:
        return float("nan"), [], None

    return luminosity_sum, fit_zones, extreme_point


def compute_luminosity_and_zones(
    gray_frame: np.ndarray,
    full_mask: np.ndarray,
    box_size: int = 21
) -> Tuple[float, List[Tuple[int, int, int]], Optional[Tuple[int, int]]]:
    """
    Fits a single 2D Gaussian centered on the extreme point along the main axis of the mask.

    Args:
        gray_frame (np.ndarray): Input grayscale frame.
        full_mask (np.ndarray): Binary mask representing the object of interest.
        box_size (int): Size of the fitting box.

    Returns:
        Tuple:
            - luminosity (float): Flux from Gaussian fit.
            - fit_zones (List[Tuple[int, int, int]]): Single bounding box for the fit.
            - extreme_point (Optional[Tuple[int, int]]): Coordinates of the extreme point.
    """
    import numpy as np
    if np.count_nonzero(full_mask) == 0:
        return 0.0, [], None

    from skimage.measure import regionprops
    import numpy as np

    props = regionprops(full_mask.astype(np.uint8))
    if len(props) == 0:
        return 0.0, [], None

    prop = props[0]
    coords = np.array(prop.coords)

    # Centre de masse
    centroid = np.mean(coords, axis=0)

    # Calcul de l'axe principal via SVD
    centered_coords = coords - centroid
    _, _, Vt = np.linalg.svd(centered_coords)
    main_axis = Vt[0]

    # Projections des points sur cet axe
    projections = centered_coords @ main_axis

    # Trouver le point extrême (le plus grand projeté)
    max_proj_idx = np.argmax(projections)
    extreme_point_coords = coords[max_proj_idx]

    y_extreme, x_extreme = extreme_point_coords

    luminosity = fit_gaussian_to_source(gray_frame, x_extreme, y_extreme, box_size=box_size, recenter=True)

    half = box_size // 2
    fit_zone = [(x_extreme - half, y_extreme - half, box_size)]
    extreme_point = (x_extreme, y_extreme)

    return luminosity, fit_zone, extreme_point
