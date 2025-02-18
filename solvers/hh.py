import numpy as np


def HH(t, y, I_app, T_stim, normalization):
    """
    Hodgkin-Huxley neuron model implementation

    Parameters:
    -----------
    t : float
        Current time
    y : array-like
        State variables [V, m, h, n]
    I_app : float
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
    C_m = 1
    g_Na = 120
    g_K = 36
    g_L = 0.3
    V_Na = 115
    V_K = -12
    V_L = 10.6

    # Gate voltage-dependent rate functions
    def alpha_m(V):
        return 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)

    def alpha_h(V):
        return 0.07 * np.exp(-V / 20)

    def alpha_n(V):
        return 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1)

    def beta_m(V):
        return 4 * np.exp(-V / 18)

    def beta_h(V):
        return 1 / (np.exp((30 - V) / 10) + 1)

    def beta_n(V):
        return 0.125 * np.exp(-V / 80)

    # State variables
    V = y[0]
    m = y[1]
    h = y[2]
    n = y[3]

    # Initialize derivatives array
    dydt = np.zeros(4)

    # Compute membrane potential derivative
    I_Na = g_Na * (m**3) * h * (V - V_Na)
    I_K = g_K * (n**4) * (V - V_K)
    I_L = g_L * (V - V_L)

    if t <= T_stim:
        dydt[0] = (-1 / C_m * (I_Na + I_K + I_L) + I_app / C_m) * normalization
    else:
        dydt[0] = (-1 / C_m * (I_Na + I_K + I_L)) * normalization

    # Compute gate variables derivatives
    dydt[1] = (alpha_m(V) * (1 - m) - beta_m(V) * m) * normalization
    dydt[2] = (alpha_h(V) * (1 - h) - beta_h(V) * h) * normalization
    dydt[3] = (alpha_n(V) * (1 - n) - beta_n(V) * n) * normalization

    return dydt
