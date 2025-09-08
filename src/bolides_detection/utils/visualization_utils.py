import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from typing import Optional, List, Tuple

def plot_three_views(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    luminosity: float,
    fit_zones: List[Tuple[int, int, int]],
    extreme_point: Optional[Tuple[int, int]] = None,
    alpha_mask: float = 0.3,
    applied_zone: Optional[np.ndarray] = None
) -> None:
    """
    Displays three views: images with overlaid mask, binary mask, and Gaussian fit zones.

    Args:
        frame (np.ndarray): Color images in BGR format.
        mask (Optional[np.ndarray]): Binary mask for overlay (can be None).
        luminosity (float): Luminosity value to display.
        fit_zones (List[Tuple[int, int, int]]): List of zones (x_min, y_min, size).
        extreme_point (Optional[Tuple[int, int]]): Coordinates (x, y) of extreme point to display.
        alpha_mask (float): Opacity of the overlaid mask.
        applied_zone (Optional[np.ndarray]): Binary mask for the model zone (overlaid in yellow).

    Returns:
        None
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Convert BGR frame to normalized RGB [0,1]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = frame_rgb.copy()

    if applied_zone is not None:
        yellow = np.array([1.0, 1.0, 0.0])  # RGB yellow
        mask_bool = applied_zone.astype(bool)
        overlay[mask_bool] = overlay[mask_bool] * (1 - 0.4) + yellow * 0.4

    if mask is not None and np.any(mask):
        norm_mask = mask.astype(np.float32)
        norm_mask /= norm_mask.max()  # Normalize for colormap
        colored_mask = plt.cm.hot(norm_mask)[..., :3]  # Colormap without alpha
        mask_bool = mask > 0
        overlay[mask_bool] = overlay[mask_bool] * (1 - alpha_mask) + colored_mask[mask_bool] * alpha_mask

    axs[0].imshow(overlay)

    axs[0].set_title("Image with mask overlay (cmap 'hot') + model zone (yellow)")
    axs[0].axis('off')

    axs[1].imshow(mask if mask is not None else np.zeros_like(frame_rgb[..., 0]), cmap='gray')
    axs[1].set_title("Binary brightness mask")
    axs[1].axis('off')

    axs[2].imshow(frame_rgb)
    for (x_min, y_min, size) in fit_zones:
        rect = patches.Rectangle((x_min, y_min), size, size,
                                 linewidth=1.5, edgecolor='cyan', facecolor='none')
        axs[2].add_patch(rect)

    if extreme_point is not None:
        axs[2].plot(extreme_point[0], extreme_point[1], 'ro', markersize=10, label='Extreme point')
        axs[2].legend()

    axs[2].set_title("Gaussian fit zones + extreme point")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_patches(patch_rgb: np.ndarray, pred_mask_patch: np.ndarray) -> None:
    """
    Displays an RGB patch and its predicted mask side-by-side.

    Args:
        patch_rgb (np.ndarray): RGB images patch.
        pred_mask_patch (np.ndarray): Predicted binary mask of the patch.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(patch_rgb)
    ax[0].set_title("Patch")
    ax[0].axis("off")
    ax[1].imshow(pred_mask_patch, cmap="hot")
    ax[1].set_title("Predicted mask")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
