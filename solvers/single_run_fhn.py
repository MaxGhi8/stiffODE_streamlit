import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(".")
from fhn import fhn
from scipy.integrate import solve_ivp


def solve_fhn(T_stim: float, I_app: float):
    # Initial conditions
    V_0 = 0
    w_0 = 0
    initial_conditions = [V_0, w_0]

    # Time parameters
    T_fin = 100
    n_points = 1260
    T_eval = np.linspace(0, T_fin, n_points)

    # Model parameters
    normalization = 1

    # Solver parameters
    atol = [1e-13, 1e-13]
    rtol = 5e-13

    # Solve ODE
    sol = solve_ivp(
        fun=fhn,
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
    sol, T_eval = solve_fhn(50, 1.5)
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(T_eval, sol.y[0])  # Plot V (first component)
    plt.xlabel("Time")
    plt.ylabel("Voltage (V)")
    plt.title("FitzHugh-Nagumo Model Simulation")
    plt.grid(True)
    plt.show()
