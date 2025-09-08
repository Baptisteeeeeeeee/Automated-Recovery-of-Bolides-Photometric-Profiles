from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from .stars_detection_utils import predict_on_full_image_with_all_points, preprocess_image
from .model import *


def detect_stars(image_path: str) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """
    Detect stars in a night sky images using a two-stage deep learning pipeline.

    Args:
        image_path (str): Path to the input images.

    Returns:
        Tuple:
            - full_mask_uint8 (np.ndarray): Binary mask (H, W) of detected star locations (0 or 1).
            - stars_px (List[Tuple[int, int]]): List of pixel coordinates (x, y) of detected stars.
            - image_data (np.ndarray): Original or cleaned images (H, W, 3) in BGR format.
    """
    # Resolve checkpoint paths
    BASE_DIR = Path(__file__).resolve().parents[3]
    segmentation_ckpt_path = BASE_DIR / "checkpoints/stars_half.pth"
    preprocessing_ckpt_path = BASE_DIR / "checkpoints/preprocess_half.pth"


    encoder = EfficientNetV2S_Encoder(variant='m')
    decoder = EfficientNetHybridDecoderM()

    # Step 1: Remove light pollution (e.g. city lights)
    cleaned_image = preprocess_image(
        image_path,
        checkpoint_path=str(preprocessing_ckpt_path),
        encoder=encoder,
        decoder=decoder
    )

    # Step 2: Predict star positions using segmentation model
    image_data, stars_px, _, full_mask = predict_on_full_image_with_all_points(
        image_path=image_path,
        img=cleaned_image,
        encoder=encoder,
        decoder=decoder,
        checkpoint_path=str(segmentation_ckpt_path)
    )

    # Convert float mask to binary uint8 mask
    full_mask_uint8 = (full_mask > 0).astype(np.uint8)

    return full_mask_uint8, stars_px, image_data
