import cv2
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any


def interactive_star_matching(
    path_original_image: str,
    image_projected_ndarray: np.ndarray,
    stars_df: pd.DataFrame,
    stars_detected: Optional[List[Tuple[int, int]]] = None
) -> Optional[pd.DataFrame]:
    """
    Interactive GUI to manually match stars between an original images and a projected images.

    Args:
        path_original_image: File path of the original images.
        image_projected_ndarray: Numpy array of the projected images.
        stars_df: DataFrame with columns ['x', 'y', 'hip'] for projected stars.
        stars_detected: Optional list of detected star coordinates (x, y) in original images.

    Returns:
        DataFrame with matched stars, including their projected and original images coordinates,
        or None if images loading fails.
    """
    try:
        img_orig = load_image(path_original_image)
    except FileNotFoundError:
        return None

    img_orig = draw_detected_stars(stars_detected, img_orig)

    img_proj = resize_to_match(image_projected_ndarray, img_orig.shape)
    combined_img, w, bar_width = create_combined_image(img_orig, img_proj)

    clicked_points = {"original": None, "projected": None}
    associations: List[Tuple[Tuple[int, int], int]] = []

    data: Dict[str, Any] = {
        'combined_img': combined_img,
        'img_orig': img_orig,
        'img_proj': img_proj,
        'w': w,
        'bar_width': bar_width,
        'clicked_points': clicked_points,
        'associations': associations,
        'stars_proj': stars_df[['x', 'y', 'hip']].copy()
    }

    cv2.namedWindow("Star Matching")
    cv2.setMouseCallback("Star Matching", unified_mouse_handler, param=data)

    redraw_combined(combined_img, img_orig, img_proj, w, bar_width)
    cv2.imshow("Star Matching", combined_img)

    while True:
        key = cv2.waitKey(20)
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()

    rows = []
    for coord_orig, idx_star in associations:
        star_row = stars_df.loc[idx_star].copy()
        star_row['x_img_orig'] = coord_orig[0]
        star_row['y_img_orig'] = coord_orig[1]
        rows.append(star_row)

    if not rows:
        # No stars identified, return empty DataFrame with expected columns
        return pd.DataFrame(columns=list(stars_df.columns) + ['x_img_orig', 'y_img_orig'])

    final_df = pd.DataFrame(rows)
    return final_df[list(stars_df.columns) + ['x_img_orig', 'y_img_orig']]


def unified_mouse_handler(event: int, x: int, y: int, flags: int, param: Dict[str, Any]) -> None:
    """
    Mouse callback handler that forwards to click and move events.

    Args:
        event: Mouse event type.
        x: X coordinate of the mouse event.
        y: Y coordinate of the mouse event.
        flags: Event flags.
        param: Dictionary with shared data.
    """
    click_event(event, x, y, flags, param)
    mouse_move_event(event, x, y, flags, param)


def click_event(event: int, x: int, y: int, flags: int, param: Dict[str, Any]) -> None:
    """
    Handles left mouse button clicks for selecting stars in original and projected images.

    Args:
        event: Mouse event type.
        x: X coordinate of the mouse event.
        y: Y coordinate of the mouse event.
        flags: Event flags.
        param: Dictionary with shared data.
    """
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    data = param
    x_proj = x - data['w'] - data['bar_width']
    clicked_points = data['clicked_points']

    if x < data['w']:
        clicked_points["original"] = (x, y)
    elif x_proj >= 0:
        clicked_points["projected"] = (x_proj, y)

    redraw_combined(data['combined_img'], data['img_orig'], data['img_proj'], data['w'], data['bar_width'])

    if clicked_points["original"]:
        cv2.circle(data['combined_img'], clicked_points["original"], 8, (0, 0, 255), 2)
    if clicked_points["projected"]:
        cv2.circle(data['combined_img'], (x, y), 8, (0, 0, 255), 2)

    cv2.imshow("Star Matching", data['combined_img'])

    if clicked_points["original"] and clicked_points["projected"]:
        associate_points(data)
        clicked_points["original"] = None
        clicked_points["projected"] = None


def mouse_move_event(event: int, x: int, y: int, flags: int, param: Dict[str, Any]) -> None:
    """
    Shows star HIP number on mouse hover over projected images stars.

    Args:
        event: Mouse event type.
        x: X coordinate of the mouse event.
        y: Y coordinate of the mouse event.
        flags: Event flags.
        param: Dictionary with shared data.
    """
    if event != cv2.EVENT_MOUSEMOVE:
        return

    data = param
    x_proj = x - data['w'] - data['bar_width']
    y_proj = y
    threshold = 10

    redraw_combined(data['combined_img'], data['img_orig'], data['img_proj'], data['w'], data['bar_width'])

    if x_proj >= 0:
        for _, star in data['stars_proj'].iterrows():
            dx = star['x'] - x_proj
            dy = star['y'] - y_proj
            if np.hypot(dx, dy) < threshold:
                hip = int(star['hip'])
                cv2.putText(data['combined_img'], f"HIP {hip}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                break

    cv2.imshow("Star Matching", data['combined_img'])


def associate_points(data: Dict[str, Any]) -> None:
    """
    Associates the clicked points from original and projected images if close enough.

    Args:
        data: Dictionary with shared data and state.
    """
    orig = data['clicked_points']["original"]
    proj = data['clicked_points']["projected"]
    threshold = 10
    min_dist = float('inf')
    best_match = None

    for idx, star in data['stars_proj'].iterrows():
        dx = star['x'] - proj[0]
        dy = star['y'] - proj[1]
        dist = np.hypot(dx, dy)
        if dist < threshold and dist < min_dist:
            best_match = (idx, star)
            min_dist = dist

    if best_match:
        idx_star, star_row = best_match
        data['associations'].append((orig, idx_star))
        # Association created, update visualization
        for (p_orig, idx_s) in data['associations']:
            star_proj = data['stars_proj'].loc[idx_s]
            p_proj_shifted = (int(star_proj['x'] + data['w'] + data['bar_width']), int(star_proj['y']))
            cv2.circle(data['combined_img'], p_orig, 8, (0, 255, 0), 2)
            cv2.circle(data['combined_img'], p_proj_shifted, 8, (0, 255, 0), 2)
            cv2.line(data['combined_img'], p_orig, p_proj_shifted, (0, 255, 0), 1)

        cv2.imshow("Star Matching", data['combined_img'])


def load_image(path: str) -> np.ndarray:
    """
    Load an images from a file.

    Args:
        path: Path to the images file.

    Returns:
        Loaded images as a NumPy array.

    Raises:
        FileNotFoundError: If the images is not found or cannot be loaded.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def resize_to_match(img_target: np.ndarray, img_reference_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize the target images to match the reference images shape.

    Args:
        img_target: Image to be resized.
        img_reference_shape: Shape of the reference images (height, width, channels).

    Returns:
        Resized images.
    """
    h, w = img_reference_shape[:2]
    return cv2.resize(img_target, (w, h))


def create_combined_image(img_orig: np.ndarray, img_proj: np.ndarray, bar_width: int = 10
                         ) -> Tuple[np.ndarray, int, int]:
    """
    Create a combined images stacking original and projected images with a separator.

    Args:
        img_orig: Original images.
        img_proj: Projected images.
        bar_width: Width of the separator bar.

    Returns:
        combined images, width of original images, width of separator bar
    """
    h, _ = img_orig.shape[:2]
    separator = 255 * np.ones((h, bar_width, 3), dtype=np.uint8)
    combined = np.hstack([img_orig, separator, img_proj])
    return combined, img_orig.shape[1], bar_width


def redraw_combined(combined_img: np.ndarray, img_orig: np.ndarray, img_proj: np.ndarray, w: int, bar_width: int) -> None:
    """
    Redraw combined images with original and projected images.

    Args:
        combined_img: Combined images array to be updated.
        img_orig: Original images.
        img_proj: Projected images.
        w: Width of original images.
        bar_width: Width of separator bar.
    """
    combined_img[:, :w, :] = img_orig
    combined_img[:, w:w + bar_width, :] = 255
    combined_img[:, w + bar_width:, :] = img_proj


def draw_detected_stars(stars_detected: Optional[List[Tuple[int, int]]], img_orig: np.ndarray) -> np.ndarray:
    """
    Draw circles on detected star coordinates on the original images.

    Args:
        stars_detected: List of star (x, y) coordinates.
        img_orig: Original images to draw on.

    Returns:
        Image with drawn detected stars.
    """
    if stars_detected is not None:
        for x, y in stars_detected:
            cv2.circle(img_orig, (int(x), int(y)), 6, (0, 255, 255), 2)
    return img_orig
