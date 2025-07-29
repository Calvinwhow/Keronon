import numpy as np

# -----------------------------
# Stop Condition Functions
# -----------------------------

def get_l1_norm(gradient_vector):
    """Calculates the L1 norm of a gradient vector."""
    return np.sum(np.abs(gradient_vector))

def check_stop_conditions(current_v, gradient_vector, l1_tolerance):
    """
    Checks if optimization should stop based on convergence or invalid values.

    Args:
        current_v (np.ndarray): Current parameter values.
        gradient_vector (np.ndarray): Current gradient vector.
        l1_tolerance (float): Tolerance for the L1 norm of the gradient.

    Returns:
        bool: True if stop conditions are met, False otherwise.
    """
    if np.isnan(current_v).any():
        print("NaN occurred in:", current_v)
        return True
    if get_l1_norm(gradient_vector) < l1_tolerance:
        print(f"Convergence detected: L1 norm of gradient < {l1_tolerance}")
        return True
    return False

def project_amp_constraints(amp_array, min_per=0.8):
    """Ensures amplitudes meet a minimum percentage constraint."""
    return np.maximum(amp_array, min_per)
