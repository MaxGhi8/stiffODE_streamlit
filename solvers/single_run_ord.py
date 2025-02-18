import matplotlib.pyplot as plt
import numpy as np
from ord import ORd
from scipy.integrate import solve_ivp


def solve_ord(T_stim: float, I_app: float):
    # Initial conditions (note: in Python we use e-5 instead of *1e-5)
    V_0 = -87.8400
    Na_i_0 = 7.23
    Na_ss_0 = 7.23
    K_i_0 = 143.79
    K_ss_0 = 143.79
    Ca_i_0 = 8.54e-5
    Ca_ss_0 = 8.43e-5
    Ca_nsr_0 = 1.61
    Ca_jsr_0 = 1.56
    m_0 = 0.0074621
    h_fast_0 = 0.692591
    h_slow_0 = 0.692574
    j_0 = 0.692477
    h_CaMK_slow_0 = 0.448501
    j_CaMK_0 = 0.692413
    m_L_0 = 0.000194015
    h_L_0 = 0.496116
    h_L_CaMK_0 = 0.265885
    a_0 = 0.00101185
    i_fast_0 = 0.999542
    i_slow_0 = 0.589579
    a_CaMK_0 = 0.000515567
    i_CaMK_fast_0 = 0.999542
    i_CaMK_slow_0 = 0.641861
    d_0 = 2.43015e-9
    f_fast_0 = 1
    f_slow_0 = 0.910671
    f_Ca_fast_0 = 1
    f_Ca_slow_0 = 0.99982
    j_Ca_0 = 0.999977
    n_0 = 0.00267171
    f_CaMK_fast_0 = 1
    f_Ca_CaMK_fast_0 = 1
    x_r_fast_0 = 8.26608e-6
    x_r_slow_0 = 0.453268
    x_s1_0 = 0.270492
    x_s2_0 = 0.0001963
    x_k1_0 = 0.996801
    J_rel_NP_0 = 2.53943e-5
    J_rel_CaMK_0 = 3.17262e-7
    CaMK_trap_0 = 0.0124065

    # Combine all initial conditions into a single array
    initial_conditions = np.array(
        [
            V_0,
            Na_i_0,
            Na_ss_0,
            K_i_0,
            K_ss_0,
            Ca_i_0,
            Ca_ss_0,
            Ca_nsr_0,
            Ca_jsr_0,
            m_0,
            h_fast_0,
            h_slow_0,
            j_0,
            h_CaMK_slow_0,
            j_CaMK_0,
            m_L_0,
            h_L_0,
            h_L_CaMK_0,
            a_0,
            i_fast_0,
            i_slow_0,
            a_CaMK_0,
            i_CaMK_fast_0,
            i_CaMK_slow_0,
            d_0,
            f_fast_0,
            f_slow_0,
            f_Ca_fast_0,
            f_Ca_slow_0,
            j_Ca_0,
            n_0,
            f_CaMK_fast_0,
            f_Ca_CaMK_fast_0,
            x_r_fast_0,
            x_r_slow_0,
            x_s1_0,
            x_s2_0,
            x_k1_0,
            J_rel_NP_0,
            J_rel_CaMK_0,
            CaMK_trap_0,
        ]
    )

    # Simulation parameters
    T_fin = 1000
    n_points = 10080

    # Create time points for evaluation
    t_eval = np.linspace(0, T_fin, n_points)

    # Define solver options (equivalent to odeset in MATLAB)
    atol = 1e-6
    rtol = 1e-6

    # Solve the ODE system
    sol = solve_ivp(
        ORd,
        (0, T_fin),
        initial_conditions,
        method="BDF",  # equivalent to ode15s
        args=(I_app, T_stim),
        t_eval=t_eval,
        atol=atol,
        rtol=rtol,
        max_step=1,
    )

    return sol, t_eval


if __name__ == "__main__":
    sol, t_eval = solve_ord(8, 50)
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, sol.y[0])
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Action Potential")
    plt.grid(True)
    plt.show()
