import numpy as np
from optimizer.target_functions import loss_function

# -----------------------------
# Gradient Optimization Functions
# -----------------------------

def partial_difference_quotient(v, loss_current, index, h, sphere_coords, L, lam, directional_models=None, weight=None):
    """Computes the partial derivative of the loss function with respect to a parameter."""
    v_forward = np.copy(v)
    v_forward[index] += h
    loss_forward = loss_function(sphere_coords, v_forward, L, lam, directional_models, weight)
    return (loss_forward - loss_current) / h


def gradient_vector_handler(v, h, sphere_coords, L, lam, directional_models=None, weight=None):
    """Computes the gradient vector for the loss function."""
    loss_current = loss_function(sphere_coords, v, L, lam, directional_models, weight)
    return np.array([
        partial_difference_quotient(v, loss_current, i, h, sphere_coords, L, lam, directional_models, weight)
        for i in range(len(v))
    ])

def gradient_ascent(gradient_vector, v, alpha):
    """Performs a gradient ascent step."""
    return v + alpha * gradient_vector
