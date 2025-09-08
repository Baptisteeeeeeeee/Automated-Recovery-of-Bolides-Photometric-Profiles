import os
from typing import Optional

import cv2

from .manual_calibration.calibration.calibration_gui import run_calibration
from .manual_calibration.manual_calibration import manual_calibration
from .luminosity_utils import compute_photometric_luminosities
import time
from .automated_calibration.automated_calibration import  automated_calibration
import pandas as pd
from .stars_detection.detect_stars import detect_stars


def calibration(image_path: str) -> pd.DataFrame:
    """
    Perform star calibration on an images using automated calibration first,
    then fallback to manual calibration if automated fails.

    Args:
        image_path: Path to the input star images.

    Returns:
        A DataFrame with matched stars and their photometric luminosities.
        Returns an empty DataFrame if calibration fails or is cancelled.
    """
    start = time.time()

    full_mask, stars_px, image = detect_stars(image_path)
    h, w = full_mask.shape
    matches =automated_calibration(full_mask,stars_px , image)
    if matches.empty:
        print("[INFO] Automated astrometry failed. Switching to manual calibration.")
        manual_result: Optional[dict] = run_calibration()

        if manual_result:
            matches = manual_calibration(
                image_path,
                manual_result["lat"],
                manual_result["lon"],
                manual_result["alt"],
                manual_result["date"],
                manual_result["hfov"],
                manual_result["vfov"],
                manual_result["az_center"],
                manual_result["alt_center"],
                full_mask
            )
        else:
            print("[INFO] Calibration cancelled by user.")

        # Compute photometric luminosities if matches found
    if not matches.empty:
        matches = compute_photometric_luminosities(image_path, matches)
        img_visu = image.copy()

        for _, row in matches.iterrows():
            x = int(row["x_detected"])
            y = int(row["y_detected"])
            mag = row["magnitude"]

            radius = max(1, int(6 - mag))
            color = (0, 255, 0)
            cv2.circle(img_visu, (x, y), radius, color, 2)

            cv2.putText(
                img_visu,
                str(row["hip"]),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        filename = os.path.basename(image_path)
        out_path = os.path.join("output", filename.replace(".jpg", "_with_stars.jpg").replace(".png", "_with_stars.png"))
        cv2.imwrite(out_path, img_visu)
        print(f"[INFO] Annotated image saved to {out_path}")


    print(f"[INFO] Calibration time: {time.time() - start:.2f} seconds")

    if not matches.empty:
        print(matches[["x_detected", "y_detected", "hip", "magnitude"]])
    else:
        print("[INFO] No stars identified.")


    return matches



if __name__ == "__main__":
    for root, dirs, files in os.walk("/Users/baptiste/PycharmProjects/PFE/data/10_Octubre_2023"):
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                calibration(file_path)

