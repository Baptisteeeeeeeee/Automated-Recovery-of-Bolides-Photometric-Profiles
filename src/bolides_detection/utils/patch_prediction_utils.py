import numpy as np
import cv2
import torch
from typing import List, Tuple
from torchvision import transforms
from .visualization_utils import plot_patches
from .model import Unet
import matplotlib.pyplot as plt

def generate_patch_coords_in_bolide_motion_zone(
    mask: np.ndarray,
    patch_size: int = 224,
    stride: int = 112
) -> List[Tuple[int, int]]:
    """
    Generate patch coordinates within a detected motion zone.
    Returns list of (y, x) tuples.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return []

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    coords = []
    for y in range(ymin, ymax - patch_size + 1, stride):
        for x in range(xmin, xmax - patch_size + 1, stride):
            coords.append((y, x))  # ✅ FIX: (y, x)

    # Handle bottom and right edges
    if (ymax - ymin) % stride != 0:
        y = ymax - patch_size + 1
        for x in range(xmin, xmax - patch_size + 1, stride):
            coords.append((y, x))
    if (xmax - xmin) % stride != 0:
        x = xmax - patch_size + 1
        for y in range(ymin, ymax - patch_size + 1, stride):
            coords.append((y, x))

    coords.append((ymax - patch_size + 1, xmax - patch_size + 1))
    return list(set(coords))



def generate_patch_coords_centered_on_motion(
    mask: np.ndarray,
    patch_size: int = 224
) -> List[Tuple[int, int]]:
    """
    Génère UNE coordonnée de patch centrée sur la zone de mouvement détectée.
    Retourne [(y, x)] — coin supérieur gauche du patch centré.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return []

    h, w = mask.shape

    # Centre du mouvement
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))

    # Coordonnée du coin supérieur gauche du patch
    x = cx - patch_size // 2
    y = cy - patch_size // 2

    # Clamp dans les limites de l'images
    x = max(0, min(x, w - patch_size))
    y = max(0, min(y, h - patch_size))

    return [(y, x)]



def load_model(checkpoint_path: str, encoder: str, decoder: str) -> Unet:
    """
    Load a U-Net model from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        encoder (str): Encoder model name.
        decoder (str): Decoder model name.

    Returns:
        Unet: Loaded model ready for inference.
    """
    model = Unet(encoder, decoder)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict_patch(patch: np.ndarray, model: Unet) -> np.ndarray:
    """
    Predict binary mask for a given images patch.

    Args:
        patch (np.ndarray): RGB images patch (H x W x 3).
        model (Unet): U-Net segmentation model.

    Returns:
        np.ndarray: Predicted binary mask (H x W) as uint8.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(patch).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()
        output = (output > 0.5).astype(np.uint8) * 255
    return output


def adjust_patch_coords(
    x: int,
    y: int,
    patch_size: int,
    img_width: int,
    img_height: int
) -> Tuple[int, int]:
    """
    Adjust patch coordinates to keep them within the images boundaries.

    Args:
        x (int): Initial x coordinate.
        y (int): Initial y coordinate.
        patch_size (int): Patch size.
        img_width (int): Image width.
        img_height (int): Image height.

    Returns:
        Tuple[int, int]: Corrected (x, y) coordinates.
    """
    x = max(0, min(x, img_width - patch_size))
    y = max(0, min(y, img_height - patch_size))
    return x, y



def predict_full_mask_from_patches(
    frame: np.ndarray,
    patch_coords: List[Tuple[int, int]],
    model: torch.nn.Module,
    patch_size: int,
    w: int,
    h: int,
    show_patches: bool = True,
    motion_mask: np.ndarray = None
) -> np.ndarray:
    """
    Predicts the full segmentation mask for a frame using patches.
    """
    model.eval()
    full_mask = np.zeros((h, w), dtype=np.uint8)

    with torch.no_grad():
        for (y, x) in patch_coords:  # ✅ FIX: Use (y, x)
            # ✅ FIX: Adjust (x, y) properly
            x, y = adjust_patch_coords(x, y, patch_size, w, h)
            print(f"[Patch] y={y}, x={x}")
            patch = frame[y:y + patch_size, x:x + patch_size]

            # Predict mask
            pred_mask = predict_patch(patch, model)

            # Insert into full_mask
            full_mask[y:y + patch_size, x:x + patch_size] = np.maximum(
                full_mask[y:y + patch_size, x:x + patch_size],
                cv2.resize(pred_mask, (patch_size, patch_size))
            )

            if show_patches:
                plt.figure(figsize=(6, 6))
                plt.imshow(patch)

                if motion_mask is not None:
                    motion_patch = motion_mask[y:y + patch_size, x:x + patch_size]
                    contours, _ = cv2.findContours(
                        motion_patch.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    for cnt in contours:
                        cnt = cnt.squeeze()
                        if cnt.ndim == 2 and cnt.shape[0] > 2:
                            plt.plot(cnt[:, 0], cnt[:, 1], color='lime', linewidth=1)

                pred_colored = plt.cm.jet(pred_mask / 255.0)
                pred_colored[..., 3] = 0.3
                plt.imshow(pred_colored)

                plt.title(f"Patch @ (x={x}, y={y})")
                plt.axis('off')
                plt.tight_layout()
                plt.pause(0.001)

    if show_patches:
        plt.show()
        cv2.destroyAllWindows()

    return full_mask


def predict_full_mask_from_motion(
    frame: np.ndarray,
    patch_coords: List[Tuple[int, int]],
    motion_mask: np.ndarray,
    patch_size: int,
    w: int,
    h: int,
    show_patches: bool = True
) -> np.ndarray:
    """
    Construit un full mask en se basant uniquement sur le mouvement détecté (motion_mask) dans chaque patch.
    """
    full_mask = np.zeros((h, w), dtype=np.uint8)

    for (y, x) in patch_coords:
        x, y = adjust_patch_coords(x, y, patch_size, w, h)

        patch_motion = motion_mask[y:y + patch_size, x:x + patch_size]

        # Critère de mouvement significatif dans le patch
        motion_score = np.sum(patch_motion > 0)
        if motion_score > 10:  # seuil ajustable
            full_mask[y:y + patch_size, x:x + patch_size] = 255

        if show_patches:
            patch_img = frame[y:y + patch_size, x:x + patch_size]
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB))

            contours, _ = cv2.findContours(
                patch_motion.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                cnt = cnt.squeeze()
                if cnt.ndim == 2 and cnt.shape[0] > 2:
                    plt.plot(cnt[:, 0], cnt[:, 1], color='lime', linewidth=1)

            plt.title(f"Patch @ (x={x}, y={y}) - Motion score: {motion_score}")
            plt.axis('off')
            plt.tight_layout()
            plt.pause(0.001)

    if show_patches:
        plt.show()
        cv2.destroyAllWindows()

    return full_mask
