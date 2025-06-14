import numpy as np
from sklearn.cluster import DBSCAN
from typing import Tuple, List, Dict
from scipy.ndimage import gaussian_filter
from optimizer.optimizer_utils import optimize_sphere_values
from vta_model.evaluate_directional_vta import EvaluateDirectionalVta

def find_clusters(coordinates: np.ndarray, epsilon: float = 2.0, max_attempts: int = 5, epsilon_increment: float = 0.5) -> Dict:
    """
    DBSCAN-based clustering algorithm that finds the largest cluster of positive points.
    Tries with an initial epsilon and increases it if no valid cluster is found.

    Args:
        coordinates (np.ndarray): An Nx4 array of [x, y, z, r] points.
        epsilon (float): Initial DBSCAN radius parameter.
        max_attempts (int): Maximum number of attempts to find a valid cluster.
        epsilon_increment (float): Increment to increase epsilon if no valid cluster is found.

    Returns:
        Average coordinate of the largest cluster
    """
    coordinates_copy = coordinates.copy()
    coordinates_copy[:, 3] = coordinates_copy[:, 3] / np.max(coordinates_copy[:, 3])
    xyzr = np.column_stack((coordinates_copy[:, :3], coordinates_copy[:, 3]))

    for attempt in range(max_attempts):
        db = DBSCAN(eps=epsilon, min_samples=2).fit(xyzr)
        labels = db.labels_
        if np.any(labels != -1):  # Check if there is any valid cluster
            largest_label = np.argmax(np.bincount(labels[labels != -1]))
            largest_cluster = coordinates_copy[labels == largest_label]
            avg = np.mean(largest_cluster, axis=0)
            return {
                "average": {"x": avg[0], "y": avg[1], "z": avg[2], "r": avg[3]},
            }
        epsilon += epsilon_increment  # Increase epsilon for the next attempt

    # Return an empty result if no valid cluster is found after all attempts
    return {
        "average": {"x": np.nan, "y": np.nan, "z": np.nan, "r": np.nan},
    }

def arctan_normalize(coords: np.ndarray) -> np.ndarray:
    normalized = coords.copy()
    normalized[:, 3] = np.arctan(normalized[:, 3])
    return normalized

def get_v(
    L: np.ndarray,
    sphere_coords: np.ndarray,
    box_size: float,
    epsilon: float = 2,
    sigma: float = 1.0
) -> Tuple[np.ndarray, Dict]:
    """Filters points inside a box centered at a given sphere coordinate."""
    r = box_size
    v = []
    for coords in sphere_coords:
        # Boolean mask for points inside the bounding box
        cx, cy, cz = coords[:3]
        distances = np.sqrt((L[:, 0] - cx)**2 + (L[:, 1] - cy)**2 + (L[:, 2] - cz)**2)
        mask = distances <= r
        filtered_L = L[mask]
        clusters = find_clusters(filtered_L, epsilon=epsilon)
        sweetspot_coord = [
            clusters['average']['x'],
            clusters['average']['y'],
            clusters['average']['z']
        ]
        distance = np.linalg.norm(np.array([cx, cy, cz]) - np.array(sweetspot_coord))
        v.append(positivity_metric(distance, clusters['average']['r']))
    total_v = sum(v)
    v = [x * (4 / total_v) for x in v]
    return np.array(v)


def positivity_metric(value: float, distance: float, k: float = 1.0, d0: float = 1.0) -> float:
    """
    Args:
        value (float): The value to be evaluated.
        distance (float): The distance to the sweetspot.
        k (float): The steepness of the sigmoid curve.
        d0 (float): The distance at which the sigmoid is centered.

    Returns:
        float: A metric representing the positivity and closeness to the target value.
    """
    # Tried some other things as well - feel free to play around with this
    # ex: return (1 / distance) * (value)
    # Sigmoid function model
    return value / (1 + np.exp(k * (distance - d0)))

def handle_nii_map( L: np.ndarray, sphere_coords: np.ndarray, directional_models: list=None):
    '''
    Params:
        directional_models: Optional list of EvaluateDirectionalVta instances (or None).
    '''
    L = arctan_normalize(L)
    v = get_v(L, sphere_coords, 10)
    print("Input V:", v)
    output_v = optimize_sphere_values( sphere_coords, v, L, directional_models)
    print("Optimized V:", output_v)
    return output_v
