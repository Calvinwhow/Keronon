import numpy as np
from optimizer.target_functions import loss_function

# -----------------------------
# Gradient Constraints
# -----------------------------

def clip_amps(v, lower_limit=0):
    """Clips amplitudes to a minimum value."""
    return np.maximum(v, lower_limit)

def clip_gradient(gradient_vector, limit=4):
    """Clips gradient values to a specified limit."""
    return np.clip(gradient_vector, -limit, limit)

# -----------------------------
# Contact Amperage constraints
# -----------------------------

def project_constraints(v, max_total=5):
    """Projects amplitudes to ensure their sum does not exceed a maximum."""
    current_sum = np.sum(v)
    if current_sum == 0:
        return np.zeros_like(v)
    else:
        return (v / current_sum) * max_total


def project_contacts(sphere_coords, v, L, lam=1, directional_models=None, weight=None):
    """
    Projects losses to ensure they are positive and within a specified range.
    
    Args:
        sphere_coords (np.ndarray): Coordinates of the spheres.
        v (np.ndarray): Current amplitudes.
        L (np.ndarray): Landscape data.
        lam (float): Regularization parameter.
        directional_models: Optional list of EvaluateDirectionalVta instances (or None).

    Returns:
        np.ndarray: Adjusted amplitudes.
    """
    total_amplitude = np.sum(v)
    losses = [
        (idx, loss_function(sphere_coords, np.eye(1, len(v), idx)[0] * val, L, lam, directional_models, weight))
        for idx, val in enumerate(v) if val > 0
    ]
    losses = [(idx, loss) for idx, loss in losses if loss >= 0]
    total_loss = sum(loss for _, loss in losses)
    if total_loss > 0:
        losses = [(idx, round((loss / total_loss) * total_amplitude, 1)) for idx, loss in losses]
        losses = [(idx, loss) for idx, loss in losses if loss > 0]
    v = np.zeros_like(v)
    for idx, amplitude in losses:
        v[idx] = amplitude
    return v
