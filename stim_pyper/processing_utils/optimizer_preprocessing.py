import numpy as np
from sklearn.cluster import DBSCAN
from typing import Tuple, List, Dict
from scipy.ndimage import gaussian_filter
from stim_pyper.optimizer.optimizer_utils import optimize_sphere_values
from stim_pyper.vta_model.evaluate_directional_vta import EvaluateDirectionalVta

### Preprocessing for L ###
def arctan_normalize(coords: np.ndarray) -> np.ndarray:
    normalized = coords.copy()
    normalized[:, 3] = np.arctan(normalized[:, 3])
    return normalized

def normalize(coords: np.ndarray) -> np.ndarray:
    """
    Normalize the coordinates by max. 
    Args:
        coords (np.ndarray): Nx4 array of coordinates.
    Returns:
        np.ndarray: Normalized coordinates.
    """     
    coords = coords.copy()
    max_val = np.max(coords[:, 3])
    coords[:, 3] = (coords[:, 3]) / (max_val)
    return coords

def bounding_box(L: np.ndarray, sphere_coords: np.ndarray = None, bbox_length: float = 30) -> np.ndarray:
    """
    Finds coordinates of L within some range of the avg contact coordinate. Then indexes L by these indices to subsample L. 
    Args:
        bbox_length (float): Scalar length in mm. Defaults to 30mm
    Returns:
        np.ndarray: Bounding box coordinates.
    """
    half_size = bbox_length / 2
    cx, cy, cz = np.mean(sphere_coords, axis=0)[:3]
    mask = (
        (L[:, 0] >= cx - half_size) & (L[:, 0] <= cx + half_size) &
        (L[:, 1] >= cy - half_size) & (L[:, 1] <= cy + half_size) &
        (L[:, 2] >= cz - half_size) & (L[:, 2] <= cz + half_size)
    )
    return L[mask]

### Initial Guess Helpers ###
def find_clusters_max(coordinates: np.ndarray, epsilon: float = 2.0) -> Dict:
    """
    Identify the cluster with the maximum radius value from a set of coordinates.

    Args:
        coordinates (np.ndarray): An Nx4 array of [x, y, z, r] points.
        epsilon (float): Initial radius parameter for clustering (unused in this function).
        max_attempts (int): Maximum number of attempts to find a valid cluster (unused in this function).
        epsilon_increment (float): Increment to increase epsilon if no valid cluster is found (unused in this function).

    Returns:
        Dict: A dictionary containing the average coordinates of the cluster with the maximum radius.
    """
    coordinates_copy = coordinates.copy()
    coordinates_copy[:, 3] = coordinates_copy[:, 3] / np.max(coordinates_copy[:, 3])
    max_index = np.argmax(coordinates[:, 3])
    max_coord = coordinates[max_index]
    return {
        "average": {"x": max_coord[0], "y": max_coord[1], "z": max_coord[2], "r": max_coord[3]},
    }


def find_clusters_dbscan(coordinates: np.ndarray, epsilon: float = 2.0, max_attempts: int = 5, epsilon_increment: float = 0.5) -> Dict:
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
        try:
            db = DBSCAN(eps=epsilon, min_samples=2).fit(xyzr)
        except Exception as e:
            print("DBSCAN failed")
            return {
                "average": {"x": np.nan, "y": np.nan, "z": np.nan, "r": np.nan},
            }
        labels = db.labels_
        if np.any(labels != -1):
            largest_label = np.argmax(np.bincount(labels[labels != -1]))
            largest_cluster = coordinates_copy[labels == largest_label]
            avg = np.mean(largest_cluster, axis=0)
            return {
                "average": {"x": avg[0], "y": avg[1], "z": avg[2], "r": avg[3]},
            }
        epsilon += epsilon_increment

    return {
        "average": {"x": np.nan, "y": np.nan, "z": np.nan, "r": np.nan},
    }

def find_clusters(L, epsilon, method):
    allowed_methods=['max', 'dbscan']
    if method not in allowed_methods:
        raise ValueError(f"Method {method} not supported. Please choose method={allowed_methods}")
    if method == 'max':
        return find_clusters_max(L, epsilon=epsilon)
    elif method == 'dbscan':        
        return find_clusters_dbscan(L, epsilon)
    else:
        return None

def positivity_metric(value: float, distance: float, d0: float = 1.0) -> float:
    """
    Returns:
        float: A metric representing the positivity and closeness to the target value.
    """
    return value / (1 + np.exp(distance - d0))

def get_v(L: np.ndarray, sphere_coords: np.ndarray, bbox_length: float, epsilon: float = 2, method: str = 'dbscan') -> Tuple[np.ndarray, Dict]:
    """Filters points inside a box centered at a given sphere coordinate."""
    r = bbox_length/2
    v = []
    for coords in sphere_coords:
        cx, cy, cz = coords[:3]
        distances = np.sqrt((L[:, 0] - cx)**2 + (L[:, 1] - cy)**2 + (L[:, 2] - cz)**2)
        mask = distances <= r
        L_masked = L[mask]
        clusters = find_clusters(L_masked, epsilon, method)
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

def preprocess_L(L, sphere_coords, bbox_length):
    '''Normalizes L and bounds the voxels for consideration around the electrode'''
    L = normalize(L)
    L = bounding_box(L, sphere_coords, bbox_length)
    return L

def get_first_guess(L, sphere_coords, bbox_length, parallel: bool = False, thread = None):
    '''Calculates upon L to find initial guess for v array prior to optimization'''
    if parallel:
        if thread == 0:
            return get_v(L, sphere_coords, bbox_length)
        else:
            v = np.abs(np.random.normal(loc=0.5, scale=0.25, size=len(sphere_coords)))
            return (v / np.sum(v)) * 4
    try:
        v = get_v(L, sphere_coords, bbox_length)
    except Exception as e:
        v = np.random.normal(loc=0.5, scale=0.1, size=len(sphere_coords))
    return v

### Orchestration Function ###
def optimize(L: np.ndarray, sphere_coords: np.ndarray, directional_models_list: List[EvaluateDirectionalVta]=None, parallel: bool = False, thread = None):
    '''Handler function to preprocess the landscape values, get initial guess for v, and call the optimization.'''
    L = preprocess_L(L, sphere_coords, bbox_length=30)
    v = get_first_guess(L, sphere_coords, bbox_length=30, parallel=parallel, thread=thread)
    return optimize_sphere_values(sphere_coords, v, L, directional_models=directional_models_list, lam=0.001)
