import matplotlib.pyplot as plt
import numpy as np
from hh import HH
from scipy.integrate import solve_ivp


def solve_hh(T_stim: float, I_app: float):
    # Initial conditions
    V_0 = 2.7570e-4
    m_0 = 5.2934e-2
    h_0 = 5.9611e-1
    n_0 = 3.1768e-1
    initial_conditions = [V_0, m_0, h_0, n_0]

    # Time parameters
    T_fin = 100
    n_points = 1260
    T_eval = np.linspace(0, T_fin, n_points)

    # Model parameters
    normalization = 1

    # Solver parameters
    atol = [1e-13, 1e-13, 1e-13, 1e-13]
    rtol = 5e-13

    # Solve ODE
    sol = solve_ivp(
        fun=HH,
        t_span=(0, T_fin),
        y0=initial_conditions,
        method="BDF",  # equivalent to ode15s
        t_eval=T_eval,
        atol=atol,
        rtol=rtol,
        args=(I_app, T_stim, normalization),
        max_step=5,
        first_step=1e-5,
    )

    return sol, T_eval


if __name__ == "__main__":
    sol, T_eval = solve_hh(8, 50)
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(T_eval, sol.y[0])  # Plot V (membrane potential)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title("Hodgkin-Huxley Model Simulation")
    plt.grid(True)
    plt.show()
