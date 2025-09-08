import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Tuple, List
from stars.stars_detection.cnn.model import Unet


def load_model(checkpoint_path: str, encoder: nn.Module, decoder: nn.Module) -> nn.Module:
    """
    Load a U-Net model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model weights (.pth file).
        encoder (nn.Module): Encoder part of the U-Net.
        decoder (nn.Module): Decoder part of the U-Net.

    Returns:
        nn.Module: Loaded U-Net model in evaluation mode.
    """
    model = Unet(encoder, decoder)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def load_model2(checkpoint_path: str, encoder: nn.Module, decoder: nn.Module) -> nn.Module:
    """
    Load a U-Net model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model weights (.pth file).
        encoder (nn.Module): Encoder part of the U-Net.
        decoder (nn.Module): Decoder part of the U-Net.

    Returns:
        nn.Module: Loaded U-Net model in evaluation mode.
    """
    model = Unet(encoder, decoder)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def predict_patch(patch: np.ndarray, model: nn.Module) -> np.ndarray:
    """
    Predict a binary heatmap mask from a given images patch.

    Args:
        patch (np.ndarray): Image patch of shape (H, W, 3), BGR format.
        model (nn.Module): Pre-trained U-Net model.

    Returns:
        np.ndarray: Binary mask of shape (H, W), with values 0 or 1.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(patch).unsqueeze(0)  # Shape: (1, 3, H, W)
    with torch.no_grad():
        output = model(tensor)
        output = torch.sigmoid(output).squeeze().numpy()  # Shape: (H, W)
        binary_mask = (output > 0.5).astype(np.uint8)
    return binary_mask


def remove_city(original_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Mask out detected bright (e.g., urban light) areas from an images.

    Args:
        original_image (np.ndarray): Original images (H, W, 3), BGR.
        mask (np.ndarray): Binary mask of detected light pollution.

    Returns:
        np.ndarray: Image with bright areas removed (masked).
    """
    h, w = original_image.shape[:2]
    mask_resized = cv2.resize(mask, (w, h))
    inverted_mask = np.logical_not(mask_resized > 0.5).astype(np.uint8)

    mask_3c = np.repeat(inverted_mask[:, :, np.newaxis], 3, axis=2)
    image_masked = original_image * mask_3c

    plt.imshow(cv2.cvtColor(image_masked, cv2.COLOR_BGR2RGB))
    plt.title("City Light Masked Image")
    plt.axis("off")
    plt.show()

    return image_masked


def preprocess_image(
    image_path: str,
    checkpoint_path: str,
    encoder: nn.Module,
    decoder: nn.Module
) -> np.ndarray:
    """
    Preprocess the input images by removing city light contamination.

    Args:
        image_path (str): Path to the input images.
        checkpoint_path (str): Path to the segmentation model weights.
        encoder (nn.Module): Encoder model.
        decoder (nn.Module): Decoder model.

    Returns:
        np.ndarray: Cleaned images with city light masked.
    """
    model = load_model(checkpoint_path, encoder, decoder)
    original_image = cv2.imread(image_path)
    resized = cv2.resize(original_image, (224, 224))
    binary_mask = predict_patch(resized, model)
    cleaned_image = remove_city(original_image, binary_mask)
    return cleaned_image


def predict_on_full_image_with_all_points(
    image_path: str,
    img: np.ndarray,
    encoder: nn.Module,
    decoder: nn.Module,
    patch_size: int = 224,
    stride: int = 224,
    checkpoint_path: str = ""
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[float], np.ndarray]:
    """
    Apply the segmentation model over the entire images in patches and extract star locations and intensities.

    Args:
        image_path (str): Path to the input images.
        img (np.ndarray): Loaded images (same as image_path, preloaded to avoid reloading).
        encoder (nn.Module): Encoder of the U-Net.
        decoder (nn.Module): Decoder of the U-Net.
        patch_size (int): Size of patches to slide over the images.
        stride (int): Stride for patch sliding (usually same as patch_size to avoid overlap).
        checkpoint_path (str): Path to the segmentation model weights.

    Returns:
        Tuple:
            orig_image (np.ndarray): Original full images (BGR).
            detected_points (List[Tuple[int, int]]): List of star (x, y) positions in images coordinates.
            intensities (List[float]): Normalized brightness values at each detected point.
            full_mask (np.ndarray): Binary segmentation mask of the full images.
    """
    orig_image = cv2.imread(image_path)
    H, W, _ = orig_image.shape
    model = load_model2(checkpoint_path, encoder, decoder)

    detected_points = []
    intensities = []
    full_mask = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img[y:y + patch_size, x:x + patch_size]
            pred_mask = predict_patch(patch, model).astype(np.float32)

            if pred_mask.max() > 1.0:
                pred_mask /= 255.0

            # Merge patch prediction into the full mask
            full_mask[y:y+patch_size, x:x+patch_size] = np.maximum(
                full_mask[y:y+patch_size, x:x+patch_size], pred_mask
            )

            # Extract connected components (i.e., star candidates)
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (pred_mask > 0.5).astype(np.uint8), connectivity=8
            )

            for i in range(1, len(centroids)):  # Skip background
                cX, cY = centroids[i]
                global_x = int(x + cX)
                global_y = int(y + cY)

                # Estimate brightness (mean mask value around centroid)
                margin = 2
                x0 = max(0, int(cX) - margin)
                x1 = min(patch_size, int(cX) + margin + 1)
                y0 = max(0, int(cY) - margin)
                y1 = min(patch_size, int(cY) + margin + 1)
                local_intensity = pred_mask[y0:y1, x0:x1].mean()

                detected_points.append((global_x, global_y))
                intensities.append(float(local_intensity))

    return orig_image, detected_points, intensities, full_mask
