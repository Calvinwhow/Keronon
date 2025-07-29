import numpy as np

# -----------------------------
# ADAM Optimization Functions
# -----------------------------

def initialize_adam(v):
    """Initializes the state for the ADAM optimizer."""
    return {
        "m": np.zeros_like(v),
        "v": np.zeros_like(v),
        "t": 0
    }

def momentum(m, gradient, beta1):
    """Computes the momentum term for ADAM."""
    return beta1 * m + (1 - beta1) * gradient

def variance(v, gradient, beta2):
    """Computes the variance term for ADAM."""
    return beta2 * v + (1 - beta2) * gradient ** 2

def moment_bias_c(m, beta1, t):
    """Corrects bias in the momentum term."""
    return m / (1 - beta1 ** t)

def var_bias_c(v, beta2, t):
    """Corrects bias in the variance term."""
    return v / (1 - beta2 ** t)

def adam_step(gradient_vector, v, adam_state, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs a single ADAM optimization step.

    Args:
        gradient_vector (np.ndarray): Gradient vector for the current step.
        v (np.ndarray): Current parameter values.
        adam_state (dict): State of the optimizer (momentum, variance, timestep).
        alpha (float): Learning rate.
        beta1 (float): Exponential decay rate for momentum.
        beta2 (float): Exponential decay rate for variance.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        tuple: Updated parameter values and updated optimizer state.
    """
    adam_state["t"] += 1
    adam_state["m"] = momentum(adam_state["m"], gradient_vector, beta1)
    adam_state["v"] = variance(adam_state["v"], gradient_vector, beta2)
    m_hat = moment_bias_c(adam_state["m"], beta1, adam_state["t"])
    v_hat = var_bias_c(adam_state["v"], beta2, adam_state["t"])
    updated_v = v + (alpha * m_hat) / (np.sqrt(v_hat) + epsilon)
    return updated_v, adam_state
