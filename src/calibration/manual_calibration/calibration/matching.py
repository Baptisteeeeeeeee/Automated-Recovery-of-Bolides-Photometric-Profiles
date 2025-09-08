import numpy as np
import pandas as pd
from typing import Tuple, Any


def find_nearest_from_dataset(
    target_point: Tuple[float, float],
    df: pd.DataFrame,
    x_col: str = 'x_img_orig',
    y_col: str = 'y_img_orig'
) -> Tuple[pd.Series, float]:
    """
    Find the nearest point in a DataFrame to a given target point.

    Args:
        target_point: Tuple of (x, y) coordinates to find nearest to.
        df: DataFrame containing points with x and y columns.
        x_col: Name of the column representing x-coordinates in df.
        y_col: Name of the column representing y-coordinates in df.

    Returns:
        A tuple containing the nearest row (as pd.Series) and the distance (float).
    """
    candidates = df[[x_col, y_col]].values
    distances = np.linalg.norm(candidates - np.array(target_point), axis=1)
    idx_min = np.argmin(distances)
    return df.iloc[idx_min], distances[idx_min]


def match_clusters_to_catalog(
    clusters: list[Tuple[float, float]],
    catalog_df: pd.DataFrame,
    max_dist: float = 15
) -> pd.DataFrame:
    """
    Match detected star clusters to catalog stars within a maximum distance.

    Args:
        clusters: List of detected star coordinates as (x, y) tuples.
        catalog_df: DataFrame of catalog stars with coordinates.
        max_dist: Maximum distance threshold for matching.

    Returns:
        DataFrame containing matched stars with detected coordinates and distance.
    """
    results = []
    for (x_det, y_det) in clusters:
        nearest_row, dist = find_nearest_from_dataset((x_det, y_det), catalog_df)
        if dist < max_dist:
            row_data = nearest_row.to_dict()
            row_data.update({
                'x_detected': x_det,
                'y_detected': y_det,
                'distance': dist
            })
            results.append(row_data)
    return pd.DataFrame(results)
