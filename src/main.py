from src.gui.interface import run
from src.calibration.calibration import calibration
from src.bolides_detection.main_bolide_luminosity import get_bolide_luminosity
from src.calibration.stars_detection.model import EfficientNetV2S_Encoder, EfficientNetHybridDecoderM
from src.bolides_detection.utils.patch_prediction_utils import load_model
from src.magnitude_utils import get_magnitude, plot_and_save_magnitudes
import os
import pandas as pd
from typing import Optional


def run_application() -> None:
    """
    Main function to run the photometric analysis application.

    Workflow:
    1. Launch GUI to get user input paths (images & video).
    2. Perform astrometric calibration (automated, then fallback to manual if needed).
    3. Run the bolide brightness detection using a pre-trained neural network.
    4. Convert luminosities to magnitudes using reference stars.
    5. Save both match data and photometric profile plot to the output folder.
    """

    output_directory = "./output"
    os.makedirs(output_directory, exist_ok=True)

    checkpoint_path = "../checkpoints/effM_Bol_reduce_pruned_half.pth"

    # Load trained model
    encoder = EfficientNetV2S_Encoder(variant='m')
    decoder = EfficientNetHybridDecoderM()
    model = load_model(checkpoint_path, encoder, decoder)

    # Launch GUI to get file paths
    results: Optional[dict] = run()
    if not results:
        print("[INFO] Application canceled by user.")
        return

    video_path = results["video_path"]
    image_path = results["image_path"]

    # Step 1: Calibrate stars in images
    matches: pd.DataFrame = calibration(image_path)
    matches_filename = os.path.join(output_directory, "matches.csv")
    matches.to_csv(matches_filename, index=False)

    # Step 2: Measure bolide brightness over video
    luminosities = get_bolide_luminosity(
        video_path, model,
        show_patches=False  # Optional visualization
    )

    magnitudes = get_magnitude(luminosities, matches)

    # Step 4: Plot and save the photometric profile
    photometric_profile_output_path = os.path.join(output_directory, "photometric_profile.jpg")
    plot_and_save_magnitudes(magnitudes, matches, output_path=photometric_profile_output_path)

    print(f"[INFO] Photometric analysis completed. Results saved to: {output_directory}")


if __name__ == "__main__":
    run_application()
