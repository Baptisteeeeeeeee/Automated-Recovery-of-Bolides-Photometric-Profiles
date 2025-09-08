from astropy.io import fits
from astropy.wcs import WCS
import os
import subprocess
import tempfile
from PIL import Image
import numpy as np


def ensure_output_dir_exists(output_dir: str):
    """
    Ensure the output directory exists; create it if it does not.
    """
    os.makedirs(output_dir, exist_ok=True)


def create_temp_image_file(image) -> str:
    """
    Save an images (PIL.Image or NumPy ndarray) to a temporary PNG file.

    Returns:
        str: Path to the temporary images file.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        temp_path = tmp_file.name

    if hasattr(image, "save"):
        image.save(temp_path)
    elif isinstance(image, np.ndarray):
        Image.fromarray(image).save(temp_path)
    else:
        raise TypeError("Image must be a PIL.Image or numpy.ndarray")

    return temp_path


def build_solve_field_command(temp_image_path: str, output_dir: str, index_dir: str) -> list:
    """
    Build the `solve-field` command with appropriate arguments.

    Returns:
        list: List of command line arguments for subprocess.
    """
    return [
        "solve-field",
        "--overwrite",
        "--verbose",
        f"--index-dir={index_dir}",
        f"--dir={output_dir}",
        "--scale-units", "degw",
        "--scale-low", "10.0",
        "--scale-high", "180.0",
        "--parity", "neg",
        "--pixel-error", "5.0",
        "-M", os.path.join(output_dir, "output.match"),
        temp_image_path
    ]


def run_solve_field(cmd: list, timeout: int, temp_image_path: str) -> subprocess.CompletedProcess:
    """
    Run the solve-field command with a timeout.

    Returns:
        subprocess.CompletedProcess: Result of subprocess execution.

    Raises:
        TimeoutError: If the command exceeds the given timeout.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.remove(temp_image_path)
        raise TimeoutError(f"⏱️ Astrometric resolution exceeded {timeout}s. Consider manual solving.")

    return result


def validate_astrometry_output(result: subprocess.CompletedProcess, temp_image_path: str, output_dir: str) -> str:
    """
    Validate that the solve-field ran successfully and the output FITS was generated.

    Returns:
        str: Path to the generated .new FITS file.

    Raises:
        RuntimeError: If solve-field failed.
        FileNotFoundError: If the expected output file is missing.
    """
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("❌ Astrometric resolution failed.")

    base = os.path.splitext(os.path.basename(temp_image_path))[0]
    fits_file = os.path.join(output_dir, base + ".new")

    if not os.path.exists(fits_file):
        contents = os.listdir(output_dir)
        raise FileNotFoundError(f"[❌] File {fits_file} not found.\nContents of {output_dir}: {contents}")

    return fits_file


def run_local_astrometry_from_image(
    image,
    output_dir: str = "./local",
    index_dir: str = "/Users/baptiste/Downloads/astrometry.net-0.97/data",
    timeout: int = 100
) -> str:
    """
    Perform local astrometry resolution from a given images using astrometry.net.

    Args:
        image: PIL.Image or NumPy array representing the images to solve.
        output_dir: Output directory to store result files.
        index_dir: Path to local astrometry.net index files.
        timeout: Timeout for the solve-field process in seconds.

    Returns:
        str: Path to the output FITS (.new) file containing WCS.
    """
    ensure_output_dir_exists(output_dir)
    temp_image_path = create_temp_image_file(image)
    cmd = build_solve_field_command(temp_image_path, output_dir, index_dir)
    result = run_solve_field(cmd, timeout, temp_image_path)
    os.remove(temp_image_path)
    fits_file = validate_astrometry_output(result, temp_image_path, output_dir)
    return fits_file


def extract_wcs_from_fits(fits_path: str):
    """
    Extract the WCS and images data from a FITS file.

    Args:
        fits_path: Path to the FITS file.

    Returns:
        tuple: (WCS object, images data as ndarray)
    """
    print(f"[INFO] Extracting WCS from {fits_path}")
    with fits.open(fits_path) as hdul:
        wcs = WCS(hdul[0].header)
        data = hdul[0].data
    return wcs, data


# Optional test block
if __name__ == "__main__":
    image_path = ""
    fits_with_wcs = run_local_astrometry_from_image(image_path)
    wcs, data = extract_wcs_from_fits(fits_with_wcs)
