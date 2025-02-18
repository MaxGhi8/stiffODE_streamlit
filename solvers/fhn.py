import numpy as np


def fhn(t, y, i_app, T_stim, normalization):
    """
    FitzHugh-Nagumo model implementation

    Parameters:
    -----------
    t : float
        Current time
    y : array-like
        State variables [V, w]
    i_app : float
        Applied current
    T_stim : float
        Stimulus duration
    normalization : float
        Normalization factor

    Returns:
    --------
    dydt : numpy.ndarray
        Time derivatives of state variables
    """
    # Parameters
    b = 5
    c = 1
    beta = 0.1
    delta = 1
    gamma = 0.25
    e = 0.1

    # State variables
    V = y[0]
    w = y[1]

    # Initialize derivatives array
    dydt = np.zeros(2)

    # Compute derivatives
    if t <= T_stim:
        dydt[0] = (b * V * (V - beta) * (delta - V) - c * w + i_app) * normalization
    else:
        dydt[0] = (b * V * (V - beta) * (delta - V) - c * w) * normalization

    dydt[1] = e * (V - gamma * w) * normalization

    return dydt
