import cv2
import numpy as np
from typing import Tuple, List

def compute_motion_map(video_path: str, max_frames: int = 300, mag_threshold: float = 1.0) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Computes a motion magnitude map from a video using dense optical flow.

    Args:
        video_path (str): Path to the input video.
        max_frames (int): Maximum number of frames to process.
        mag_threshold (float): Threshold below which motion magnitude is ignored.

    Returns:
        Tuple containing:
            - motion_magnitude_sum (np.ndarray): Accumulated motion magnitudes.
            - all_frames (List[np.ndarray]): List of all frames read.
    """


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file : {video_path}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Unable to read the first frame of the video.")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    motion_magnitude_sum = np.zeros((h, w), dtype=np.float32)
    all_frames = [frame]

    idx = 0
    while ret and idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=30,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag[mag < mag_threshold] = 0
        motion_magnitude_sum += mag

        prev_gray = gray.copy()
        all_frames.append(frame)
        idx += 1

    cap.release()

    return motion_magnitude_sum, all_frames



def get_moving_zones(motion_magnitude_sum: np.ndarray,
                     soft_threshold: int = 5) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identifies motion zones in a motion magnitude map using Otsu and soft thresholding.

    Args:
        motion_magnitude_sum (np.ndarray): Accumulated motion magnitude map.
        soft_threshold (int): Fixed threshold value for soft detection.

    Returns:
        Tuple containing:
            - num_labels (int): Number of connected components (excluding background).
            - labels (np.ndarray): Label matrix of connected components.
            - stats (np.ndarray): Statistics on connected components (bounding boxes, areas, etc.).
            - hard_mask (np.ndarray): Binary mask using Otsu thresholding (high confidence).
            - soft_mask (np.ndarray): Binary mask using fixed soft threshold (low confidence).
    """
    # Normalize to 0–255 range for thresholding
    motion_norm = cv2.normalize(motion_magnitude_sum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # High-confidence binary mask using Otsu's threshold
    _, hard_mask = cv2.threshold(motion_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Low-confidence mask using a fixed soft threshold
    _, soft_mask = cv2.threshold(motion_norm, soft_threshold, 255, cv2.THRESH_BINARY)

    # Extract connected components from hard mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hard_mask, connectivity=8)

    return num_labels, labels, stats, hard_mask, soft_mask


def select_most_likely_bolide_zone(num_labels: int,
                                   labels: np.ndarray,
                                   motion_magnitude_sum: np.ndarray,
                                   motion_thresh: np.ndarray,
                                   motion_soft: np.ndarray,
                                   min_motion_sum: float = 10.0) -> np.ndarray:
    """
    Selects and grows the most probable bolide motion zone based on motion magnitude.

    Args:
        num_labels (int): Number of detected connected components.
        labels (np.ndarray): Labeled regions from connected components.
        motion_magnitude_sum (np.ndarray): Accumulated motion magnitude map.
        motion_thresh (np.ndarray): Hard binary mask (e.g., from Otsu).
        motion_soft (np.ndarray): Soft mask used for growing region.
        min_motion_sum (float): Minimum motion magnitude to consider a region.

    Returns:
        np.ndarray: Final binary mask of the selected and grown bolide zone.
    """
    h, w = motion_thresh.shape
    best_label = None
    max_score = 0.0

    # Find the label with highest motion sum
    for label_id in range(1, num_labels):
        region_mask = (labels == label_id)
        region_motion_sum = np.sum(motion_magnitude_sum[region_mask])
        if region_motion_sum > max_score and region_motion_sum > min_motion_sum:
            best_label = label_id
            max_score = region_motion_sum

    final_mask = np.zeros_like(motion_thresh, dtype=np.uint8)

    if best_label is not None:
        seed_mask = (labels == best_label).astype(np.uint8)
        final_mask = _region_grow(seed_mask, motion_soft)

    return final_mask


def _region_grow(
        seed_mask: np.ndarray,
        motion_soft:
        np.ndarray
) -> np.ndarray:
    """
    Grows a region from a seed mask using a soft motion mask as constraint.

    Args:
        seed_mask (np.ndarray): Binary seed mask.
        motion_soft (np.ndarray): Soft binary motion mask.

    Returns:
        np.ndarray: Grown region mask.
    """
    h, w = seed_mask.shape
    visited = np.zeros_like(seed_mask, dtype=np.uint8)
    stack = list(zip(*np.where(seed_mask > 0)))

    while stack:
        y, x = stack.pop()
        if visited[y, x] or motion_soft[y, x] == 0:
            continue
        visited[y, x] = 255
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    if motion_soft[ny, nx] > 0:
                        stack.append((ny, nx))

    return visited

def detect_bolide_motion_zone(
    video_path: str,
    max_frames: int = 300,
    mag_threshold: float = 1.0,
    min_motion_sum: float = 10.0
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Detects the most likely bolide (fast-moving object) motion zone in a video.

    Args:
        video_path (str): Path to the input video file.
        max_frames (int): Maximum number of frames to process.
        mag_threshold (float): Minimum motion magnitude per pixel to consider.
        min_motion_sum (float): Minimum total motion in a region to be considered valid.

    Returns:
        final_mask (np.ndarray): Binary mask of the detected bolide zone.
        all_frames (List[np.ndarray]): All read frames from the video (for visualization).
    """
    motion_magnitude_sum, all_frames = compute_motion_map(
        video_path, max_frames=max_frames, mag_threshold=mag_threshold)

    if motion_magnitude_sum is None or len(all_frames) == 0:
        raise RuntimeError("Could not compute motion map. Video might be unreadable.")

    num_labels, labels, stats, motion_thresh, motion_soft = get_moving_zones(motion_magnitude_sum)

    final_mask = select_most_likely_bolide_zone(
        num_labels, labels, motion_magnitude_sum, motion_thresh, motion_soft, min_motion_sum)

    return final_mask, all_frames


def detect_motion_start_in_bolide_zone(
    all_frames: List[np.ndarray],
    final_mask: np.ndarray,
    threshold: int = 30
) -> Tuple[np.ndarray, int]:
    """
    Detect the frame index where motion starts in the detected bolide zone.

    Args:
        all_frames (List[np.ndarray]): List of video frames.
        final_mask (np.ndarray): Binary mask for the bolide zone.
        threshold (int): Pixel difference threshold to consider motion.

    Returns:
        motion_array (np.ndarray): Array of binary motion maps per frame.
        start_apply_frame (int): Frame index to start applying predictions.
    """
    motion_history = []

    for i in range(len(all_frames) - 1):
        prev_g = cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2GRAY)
        curr_g = cv2.cvtColor(all_frames[i + 1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_g, curr_g)
        _, motion_mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
        motion_history.append(motion_mask)

    motion_array = np.stack(motion_history, axis=0)

    motion_in_zone = np.any(motion_array[:, final_mask.astype(bool)], axis=1)

    if np.any(motion_in_zone):
        motion_start_frame = int(np.argmax(motion_in_zone))
        start_apply_frame = max(0, motion_start_frame - 1)
    else:
        start_apply_frame = len(motion_array) + 1

    return motion_array, start_apply_frame


def should_stop_prediction(
    motion_array: np.ndarray,
    final_mask: np.ndarray,
    f_idx: int,
    no_motion_count: int,
    max_no_motion_frames: int = 5
) -> Tuple[bool, int]:
    """
    Decide whether to stop tracking/prediction based on absence of motion.

    Args:
        motion_array (np.ndarray): Stack of binary motion maps.
        final_mask (np.ndarray): Binary mask for the bolide zone.
        f_idx (int): Current frame index.
        no_motion_count (int): Current consecutive no-motion count.
        max_no_motion_frames (int): Allowed consecutive frames with no motion before stopping.

    Returns:
        stop (bool): Whether to stop prediction.
        updated_no_motion_count (int): Updated no-motion frame counter.
    """
    if not np.any(motion_array[f_idx][final_mask.astype(bool)]):
        no_motion_count += 1
        if no_motion_count >= max_no_motion_frames:
            return True, no_motion_count
    else:
        no_motion_count = 0

    return False, no_motion_count


