import numpy as np
from stim_pyper.optimizer.adam import initialize_adam, adam_step
from stim_pyper.optimizer.gradient_ascent import gradient_vector_handler
from stim_pyper.optimizer.stop_conditions import check_stop_conditions
from stim_pyper.optimizer.constraints import clip_amps, clip_gradient, project_contacts

# -----------------------------
# Input Validation
# -----------------------------

def validate_inputs(sphere_coords, v, L):
    """Validates the inputs for the optimization process."""
    if not isinstance(sphere_coords, np.ndarray) or not isinstance(v, np.ndarray) or not isinstance(L, np.ndarray):
        raise ValueError("Inputs must be numpy arrays.")
    if sphere_coords.shape[0] != v.shape[0]:
        raise ValueError("sphereCoords and v must have the same length.")
    if len(v) == 0 or len(L) == 0 or len(sphere_coords) == 0:
        if len(v) == 0:
            print("v is empty")
        if len(L) == 0:
            print("L is empty")
        if len(sphere_coords) == 0:
            print("sphere_coords is empty")
        raise ValueError("Input arrays must not be empty.")
    if not np.any(v > 0.1):
        raise ValueError("At least one contact value in v must be greater than 0.1.")
    if np.sum(v) > 5:
        raise ValueError("Total amperage exceeds 5 mA.")
    if not all(coord.shape == (3,) for coord in sphere_coords):
        raise ValueError("Each sphere coordinate must be a 3-element array.")
    if L.shape[1] != 4:
        raise ValueError("Each landscape point must have four values: [x, y, z, magnitude]")

# -----------------------------
# Optimizer
# -----------------------------

def optimize_sphere_values(sphere_coords, 
                           v, 
                           L, 
                           directional_models=None, 
                           lam=0.0001, 
                           alpha=0.001, 
                           h=0.001, 
                           l1_tolerance=0.001):
    """
    Optimizes sphere amplitudes using gradient ascent and ADAM optimization.

    Args:
        sphere_coords (np.ndarray): Coordinates of the spheres.
        v (np.ndarray): Initial amplitudes.
        L (np.ndarray): Landscape data.
        lam (float): Regularization parameter.
        alpha (float): Learning rate.
        h (float): Step size for finite differences.
        l1_tolerance (float): Convergence tolerance for the L1 norm of the gradient.
        directional_models: Optional list of EvaluateDirectionalVta instances (or None). 
            Each instance within this list corresponds to the directional model for 
            that same index in the v array. So for any contact, it will evaluate that specific contact
            within the confines of the EvaluateDirectionalVta instance. these instances require:
            contact_coordinates, radius_mm, primary_idx=0. Contact coordinates is a list of coordinate lists. 
            If more than one sub-list (coordinate triplet) in that list, will generate a directional VTA. 
            Otherwise, will generate a spherical VTA. 

    Returns:
        np.ndarray: Optimized amplitudes.
    """
    validate_inputs(sphere_coords, v, L)
    current_v = np.copy(v)
    iteration = 0
    adam_state = initialize_adam(v)
    while iteration < 100:
        gradient_vector         = gradient_vector_handler(current_v, h, sphere_coords, L, lam, directional_models)
        clipped_gradient        = clip_gradient(gradient_vector)
        current_v, adam_state   = adam_step(clipped_gradient, current_v, adam_state, alpha)
        current_v               = clip_amps(current_v)
        if check_stop_conditions(current_v, gradient_vector, l1_tolerance):
            break
        iteration += 1
    optimized_v                 = project_contacts(sphere_coords, current_v, L, lam, directional_models)
    return optimized_v