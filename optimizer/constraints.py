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

def project_num_contacts(sphere_coords, v, N_contacts, L, lam=.1, directional_models=None):
    """
    Projects amplitudes to a specified number of active contacts.
    Isolates specific contacts, calculates their loss, ranks them, 
    and chooses to project the top N contacts, to a minimum biologically
    effective amperage. 

    Args:
        sphere_coords (np.ndarray): Coordinates of the spheres.
        v (np.ndarray): Current amplitudes.
        N_contacts (int): Number of active contacts to project.
        L (np.ndarray): Landscape data.
        lam (float): Regularization parameter.
        directional_models: Optional list of EvaluateDirectionalVta instances (or None).

    Returns:
        np.ndarray: Adjusted amplitudes with the specified number of active contacts.
        
    Note:
        This calculates the loss per contact, prioritizing the contacts which 
    """
    losses = [
        (idx, loss_function(sphere_coords, np.eye(1, len(v), idx)[0] * val, L, lam, directional_models))
        for idx, val in enumerate(v) if val > 0
    ]
    selected_indices = [idx for idx, _ in sorted(losses, key=lambda x: x[1], reverse=True)[:N_contacts]]
    mask = np.isin(np.arange(len(v)), selected_indices)
    return np.maximum(v, 1) * mask
