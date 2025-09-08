import os
from typing import List, Tuple, Optional
import torch.nn as nn
import cv2
import numpy as np
from .utils import *

def get_bolide_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute the centroid of the bolide in a binary mask.
    Returns None if no bolide detected.
    """
    moments = cv2.moments((mask > 0.5).astype(np.uint8))
    if moments["m00"] == 0:
        return None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return (cx, cy)

def get_displacement(prev_centroid, curr_centroid):
    """
    Compute displacement vector between two centroids.
    """
    if prev_centroid is None or curr_centroid is None:
        return np.array([0.0, 0.0])
    dx = curr_centroid[0] - prev_centroid[0]
    dy = curr_centroid[1] - prev_centroid[1]
    return np.array([dx, dy])

def get_bolide_luminosity(
    video_path: str,
    model: nn.Module,
    max_frames: int = 300,
    mag_threshold: float = 1.0,
    min_motion_sum: float = 10,
    patch_size: int = 224,
    stride: int = 112,
    show_patches: bool = False,
    motion_threshold: float = 0.5
) -> List[float]:
    """
    Computes bolide luminosity, stopping if the bolide recedes or stops.
    """

    # Detect the bolide motion zone and load frames
    final_mask, all_frames = detect_bolide_motion_zone(video_path, max_frames, mag_threshold, min_motion_sum)
    h, w = final_mask.shape

    # Generate patch coordinates inside detected zone
    patch_coords = generate_patch_coords_in_bolide_motion_zone(final_mask.astype(bool), patch_size=patch_size, stride=stride)

    # Detect motion start frame within bolide zone
    motion_array, start_apply_frame = detect_motion_start_in_bolide_zone(all_frames, final_mask)

    luminosity_list: List[float] = []
    prev_centroid = None
    init_vector = None

    for frame_idx in range(len(motion_array)):
        if frame_idx < start_apply_frame:
            luminosity_list.append(0.0)
            continue

        frame = all_frames[frame_idx + 1].copy()

        # Predict full mask from patches
        full_mask = predict_full_mask_from_patches(frame, patch_coords, model, patch_size, w, h, show_patches)

        # Compute centroid of the bolide in the current frame
        curr_centroid = get_bolide_centroid(full_mask)

        # Initialize the motion vector
        if prev_centroid is not None and init_vector is None:
            init_vector = get_displacement(prev_centroid, curr_centroid)
            if np.linalg.norm(init_vector) < motion_threshold:
                init_vector = None  # Ignore very small initial movement

        # Check if the bolide stopped or recedes
        if prev_centroid is not None and init_vector is not None and curr_centroid is not None:
            displacement = get_displacement(prev_centroid, curr_centroid)

            # Stop if bolide recedes
            if np.dot(displacement, init_vector) < 0:
                break

        prev_centroid = curr_centroid

        # Compute luminosity using Gaussian fitting
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luminosity, fit_zones, extreme_point = compute_luminosity_and_zones(gray_frame, full_mask)

        luminosity_list.append(luminosity)

        # Visualize results
        plot_three_views(
            frame,
            full_mask,
            luminosity,
            fit_zones,
            extreme_point=extreme_point,
            applied_zone=final_mask.astype(bool)
        )

    return luminosity_list


def read_all_videos(directory: str):
    encoder, decoder = EfficientNetV2S_Encoder(variant="m"), EfficientNetHybridDecoderM()
    model = load_model(
        "/Users/baptiste/PycharmProjects/PFE/stars/stars_detection/cnn/checkpoints_clean/effM_Bol_after_reduce.pth",
        encoder,
        decoder
    )
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4') and "br" in file:
                file_path = os.path.join(root, file)
                get_bolide_luminosity(file_path, model, show_patches=False)


if __name__ == '__main__':
    read_all_videos("/Users/baptiste/PycharmProjects/PFE/data/10_Octubre_2023")
