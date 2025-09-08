import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Union

def get_magnitude(luminosities: List[float], matches: pd.DataFrame) -> List[float]:
    """
    Computes the bolide's magnitude for each frame using reference stars.

    Args:
        luminosities (List[float]): Measured flux values for the bolide (one per frame).
        matches (pd.DataFrame): DataFrame with reference stars. Must contain:
            - 'lum_fixed': Estimated flux (from photometry)
            - 'magnitude': Known visual magnitude of the star

    Returns:
        List[float or np.nan]: Estimated bolide magnitude per frame. NaN if flux is invalid.
    """
    ref_fluxes = matches["lum_fixed"].values
    ref_mags = matches["magnitude"].values

    # Avoid division by zero
    ref_fluxes = np.clip(ref_fluxes, 1e-6, None)

    # Average flux and magnitude of reference stars
    F_ref = np.mean(ref_fluxes)
    m_ref = np.mean(ref_mags)

    magnitudes = []
    for F in luminosities:
        if F <= 0:
            magnitudes.append(np.nan)
        else:
            mag = m_ref - 2.5 * np.log10(F / F_ref)
            magnitudes.append(mag)

    return magnitudes


def plot_and_save_magnitudes(
    magnitudes: List[float],
    matches: pd.DataFrame,
    output_path: str = ""
) -> None:
    """
    Plots and saves the bolide's photometric profile (brightness over time).

    Args:
        magnitudes (List[float or np.nan]): Bolide magnitudes per frame.
        matches (pd.DataFrame): (Unused here, but can be extended for metadata).
        output_path (str): File path to save the resulting plot.
    """
    magnitudes = np.array(magnitudes)

    # Filter out NaN and zero magnitudes
    valid_indices = (~np.isnan(magnitudes)) & (magnitudes != 0)
    x = np.arange(len(magnitudes))
    x_valid = x[valid_indices]
    magnitudes_valid = magnitudes[valid_indices]

    plt.figure(figsize=(8, 5))
    plt.plot(x_valid, magnitudes_valid, marker='o', linestyle='-', color='blue')
    plt.gca().invert_yaxis()  # Lower magnitude = brighter
    plt.xlabel("Frame Index")
    plt.ylabel("Magnitude")
    plt.title("Bolide Photometric Profile")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Photometric graph saved at: {output_path}")
