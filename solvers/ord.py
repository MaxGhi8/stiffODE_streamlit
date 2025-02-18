import numpy as np


def ORd(t, y, I_app, T_stim):
    # Unpack state variables
    V = y[0]  # Note: Python uses 0-based indexing
    Na_i = y[1]
    Na_ss = y[2]
    K_i = y[3]
    K_ss = y[4]
    Ca_i = y[5]
    Ca_ss = y[6]
    Ca_nsr = y[7]
    Ca_jsr = y[8]
    m = y[9]
    h_fast = y[10]
    h_slow = y[11]
    j = y[12]
    h_CaMK_fast = h_fast  # This is computed from h_fast, not directly from y
    h_CaMK_slow = y[13]
    j_CaMK = y[14]
    m_L = y[15]
    h_L = y[16]
    h_L_CaMK = y[17]
    a = y[18]
    i_fast = y[19]
    i_slow = y[20]
    a_CaMK = y[21]
    i_CaMK_fast = y[22]
    i_CaMK_slow = y[23]
    d = y[24]
    f_fast = y[25]
    f_slow = y[26]
    f_Ca_fast = y[27]
    f_Ca_slow = y[28]
    j_Ca = y[29]
    n = y[30]
    f_CaMK_fast = y[31]
    f_Ca_CaMK_fast = y[32]
    x_r_fast = y[33]
    x_r_slow = y[34]
    x_s1 = y[35]
    x_s2 = y[36]
    x_K1 = y[37]
    J_rel_NP = y[38]
    J_rel_CaMK = y[39]
    CaMK_trap = y[40]

    R = 8314
    T = 310
    F = 96485
    Na_o = 140
    K_o = 5.4
    Ca_o = 1.8
    C_m = 1
    L = 0.01
    r = 0.0011
    v_cell = 1000 * np.pi * r * r * L
    A_geo = 2 * np.pi * r * r + 2 * np.pi * r * L
    A_cap = 2 * A_geo
    v_myo = 0.68 * v_cell
    v_nsr = 0.0552 * v_cell
    v_jsr = 0.0048 * v_cell
    v_ss = 0.02 * v_cell
    alpha_CaMK = 0.05
    beta_CaMK = 0.00068
    CaMK_0 = 0.05
    K_mCaM = 0.0015
    K_mCaMK = 0.15
    CaMK_bound = CaMK_0 * (1 - CaMK_trap) / (1 + K_mCaM / Ca_ss)
    CaMK_active = CaMK_bound + CaMK_trap
    V_Na = (R * T / F) * np.log(Na_o / Na_i)
    m_inf = 1 / (1 + np.exp(-(V + 39.57) / 9.871))
    tau_m = 1 / (
        6.765 * np.exp((V + 11.64) / 34.77) + 8.552 * np.exp(-(V + 77.42) / 5.955)
    )
    h_inf = 1 / (1 + np.exp((V + 82.9) / 6.086))
    tau_h_fast = 1 / (
        1.432 * 1e-5 * np.exp(-(V + 1.196) / 6.285)
        + 6.149 * np.exp((V + 0.5096) / 20.27)
    )
    tau_h_slow = 1 / (
        0.009794 * np.exp(-(V + 17.95) / 28.05) + 0.3343 * np.exp((V + 5.730) / 56.66)
    )
    A_h_fast = 0.99
    A_h_slow = 0.01
    h = A_h_fast * h_fast + A_h_slow * h_slow
    j_inf = h_inf
    tau_j = 2.038 + 1 / (
        0.02136 * np.exp(-(V + 100.6) / 8.281) + 0.3052 * np.exp((V + 0.9941) / 38.45)
    )
    h_CaMK_inf = 1 / (1 + np.exp((V + 89.1) / 6.086))
    tau_h_CaMK_slow = 3.0 * tau_h_slow
    A_h_CaMK_fast = A_h_fast
    A_h_CaMK_slow = A_h_slow
    h_CaMK = A_h_CaMK_fast * h_CaMK_fast + A_h_CaMK_slow * h_CaMK_slow
    j_CaMK_inf = j_inf
    tau_j_CaMK = 1.46 * tau_j
    K_m_CaMK = 0.15
    Phi_INa_CaMK = 1 / (1 + K_m_CaMK / CaMK_active)
    G_Na_fast_bar = 75
    I_Na_fast = (
        G_Na_fast_bar
        * (V - V_Na)
        * m**3
        * ((1 - Phi_INa_CaMK) * h * j + Phi_INa_CaMK * h_CaMK * j_CaMK)
    )
    m_L_inf = 1 / (1 + np.exp(-(V + 42.85) / 5.264))
    tau_m_L = tau_m
    h_L_inf = 1 / (1 + np.exp((V + 87.61) / 7.488))
    tau_h_L = 200
    h_L_CaMK_inf = 1 / (1 + np.exp((V + 93.81) / 7.488))
    tau_h_L_CaMK = 3 * tau_h_L
    K_m_CaMK = 0.15
    Phi_INaL_CaMK = 1 / (1 + (K_m_CaMK / CaMK_active))
    G_Na_late_bar = 0.0075
    I_Na_late = (
        G_Na_late_bar
        * (V - V_Na)
        * m_L
        * ((1 - Phi_INaL_CaMK) * h_L + Phi_INaL_CaMK * h_L_CaMK)
    )
    I_Na = I_Na_late + I_Na_fast
    V_K = (R * T / F) * np.log(K_o / K_i)
    alpha_inf = 1 / (1 + np.exp(-(V - 14.34) / 14.82))
    tau_a = 1.0515 / (
        1 / (1.2089 * (1 + np.exp(-(V - 18.41) / 29.38)))
        + 3.5 / (1 + np.exp((V + 100) / 29.38))
    )
    i_inf = 1 / (1 + np.exp((V + 43.94) / 5.711))
    tau_i_fast = 4.562 + 1 / (
        0.3933 * np.exp(-(V + 100) / 100) + 0.08004 * np.exp((V + 50) / 16.59)
    )
    tau_i_slow = 23.62 + 1 / (
        0.001416 * np.exp(-(V + 96.52) / 59.05)
        + 1.780 * 1e-8 * np.exp((V + 114.1) / 8.079)
    )
    A_i_fast = 1 / (1 + np.exp((V - 213.6) / 151.2))
    A_i_slow = 1 - A_i_fast
    i = A_i_fast * i_fast + A_i_slow * i_slow
    a_CaMK_inf = 1 / (1.0 + np.exp((-(V - 24.34)) / 14.82))
    tau_a_CaMK = tau_a
    i_CaMK_inf = i_inf
    delta_CaMK_develop = 1.354 + 1e-4 / (
        np.exp((V - 167.4) / 15.89) + np.exp(-(V - 12.23) / 0.2154)
    )
    delta_CaMK_recover = 1 - 0.5 / (1 + np.exp((V + 70) / 20))
    tau_i_CaMK_fast = tau_i_fast * delta_CaMK_develop * delta_CaMK_recover
    tau_i_CaMK_slow = tau_i_slow * delta_CaMK_develop * delta_CaMK_recover
    A_i_CaMK_fast = A_i_fast
    A_i_CaMK_slow = A_i_slow
    i_CaMK = A_i_CaMK_fast * i_CaMK_fast + A_i_CaMK_slow * i_CaMK_slow
    K_m_CaMK = 0.15
    Phi_Ito_CaMK = 1 / (1 + (K_m_CaMK / CaMK_active))
    G_to_bar = 0.02
    I_to = (
        G_to_bar
        * (V - V_K)
        * ((1 - Phi_Ito_CaMK) * a * i + Phi_Ito_CaMK * a_CaMK * i_CaMK)
    )
    d_inf = 1 / (1 + np.exp(-(V + 3.940) / 4.230))
    tau_d = 0.6 + 1 / (np.exp(-0.05 * (V + 6)) + np.exp(0.09 * (V + 14)))
    f_inf = 1 / (1 + np.exp((V + 19.58) / 3.696))
    tau_f_fast = 7 + 1 / (
        0.0045 * np.exp(-(V + 20) / 10) + 0.0045 * np.exp((V + 20) / 10)
    )
    tau_f_slow = 1000 + 1 / (
        0.000035 * np.exp(-(V + 5) / 4) + 0.000035 * np.exp((V + 5) / 6)
    )
    A_f_fast = 0.6
    A_f_slow = 1 - A_f_fast
    f = A_f_fast * f_fast + A_f_slow * f_slow
    f_Ca_inf = f_inf
    tau_f_Ca_fast = 7.0 + 1 / (0.04 * np.exp(-(V - 4) / 7) + 0.04 * np.exp((V - 4) / 7))
    tau_f_Ca_slow = 100 + 1 / (0.00012 * np.exp(-V / 3) + 0.00012 * np.exp(V / 7))
    A_f_Ca_fast = 0.3 + 0.6 / (1 + np.exp((V - 10) / 10))
    A_f_Ca_slow = 1 - A_f_Ca_fast
    f_Ca = A_f_Ca_slow * f_Ca_slow + A_f_Ca_fast * f_Ca_fast
    j_Ca_inf = f_Ca_inf
    tau_j_Ca = 75.0
    f_CaMK_inf = f_inf
    tau_f_CaMK_fast = 2.5 * tau_f_fast
    A_f_CaMK_fast = A_f_fast
    A_f_CaMK_slow = A_f_slow
    f_CaMK_slow = f_slow
    f_CaMK = A_f_CaMK_fast * f_CaMK_fast + A_f_CaMK_slow * f_CaMK_slow
    f_Ca_CaMK_inf = f_inf
    tau_f_Ca_CaMK_fast = 2.5 * tau_f_Ca_fast
    A_f_Ca_CaMK_fast = A_f_Ca_fast
    A_f_Ca_CaMK_slow = A_f_Ca_slow
    f_Ca_CaMK_slow = f_Ca_slow
    f_Ca_CaMK = A_f_Ca_CaMK_fast * f_Ca_CaMK_fast + A_f_Ca_CaMK_slow * f_Ca_CaMK_slow
    K_m_n = 0.002
    k_p2_n = 1000
    k_m2_n = j_Ca * 1
    alpha_n = 1 / (k_p2_n / k_m2_n + (1 + K_m_n / Ca_ss) ** 4)
    P_Ca = 0.0001
    gamma_Cai = 1
    gamma_Cao = 0.341
    z_Ca = 2
    Psi_Ca = (
        z_Ca**2
        * (V * (F**2))
        / (R * T)
        * (gamma_Cai * Ca_ss * np.exp(z_Ca * V * F / (R * T)) - gamma_Cao * Ca_o)
        / (np.exp(z_Ca * V * F / (R * T)) - 1)
    )
    I_CaL_bar = P_Ca * Psi_Ca
    P_CaNa = 0.00125 * P_Ca
    gamma_Nai = 0.75
    gamma_Nao = 0.75
    z_Na = 1
    Psi_CaNa = (
        z_Na**2
        * (V * (F**2) / (R * T))
        * (gamma_Nai * Na_ss * np.exp(z_Na * V * F / (R * T)) - gamma_Nao * Na_o)
        / (np.exp(z_Na * V * F / (R * T)) - 1)
    )
    I_CaNa_bar = P_CaNa * Psi_CaNa
    P_CaK = 3.574 * 1e-4 * P_Ca
    gamma_Ki = 0.75
    gamma_Ko = 0.75
    z_K = 1
    Psi_CaK = (
        z_K**2
        * (V * F**2)
        / (R * T)
        * (gamma_Ki * K_ss * np.exp((z_K * V * F) / (R * T)) - gamma_Ko * K_o)
        / (np.exp(z_K * V * F / (R * T)) - 1)
    )
    I_CaK_bar = P_CaK * Psi_CaK
    P_Ca_CaMK = 1.1 * P_Ca
    I_CaL_CaMK_bar = P_Ca_CaMK * Psi_Ca
    P_CaNa_CaMK = 0.00125 * P_Ca_CaMK
    I_CaNa_CaMK_bar = P_CaNa_CaMK * Psi_CaNa
    P_CaK_CaMK = 3.574 * 1e-4 * P_Ca_CaMK
    I_CaK_CaMK_bar = P_CaK_CaMK * Psi_CaK
    K_m_CaMK = 0.15
    Phi_ICaL_CaMK = 1 / (1 + K_m_CaMK / CaMK_active)
    I_CaL = I_CaL_bar * d * (1 - Phi_ICaL_CaMK) * (
        f * (1 - n) + f_Ca * n * j_Ca
    ) + I_CaL_CaMK_bar * d * Phi_ICaL_CaMK * (f_CaMK * (1 - n) + f_Ca_CaMK * n * j_Ca)
    I_CaNa = I_CaNa_bar * d * (1 - Phi_ICaL_CaMK) * (
        f * (1 - n) + f_Ca * n * j_Ca
    ) + I_CaNa_CaMK_bar * d * Phi_ICaL_CaMK * (f_CaMK * (1 - n) + f_Ca_CaMK * n * j_Ca)
    I_CaK = I_CaK_bar * d * (1 - Phi_ICaL_CaMK) * (
        f * (1 - n) + f_Ca * n * j_Ca
    ) + I_CaK_CaMK_bar * d * Phi_ICaL_CaMK * (f_CaMK * (1 - n) + f_Ca_CaMK * n * j_Ca)
    x_r_inf = 1 / (1 + np.exp(-(V + 8.337) / 6.789))
    tau_xr_fast = 12.98 + 1 / (
        0.3652 * np.exp((V - 31.66) / 3.869)
        + 4.123 * 1e-5 * np.exp(-(V - 47.78) / 20.38)
    )
    tau_xr_slow = 1.865 + 1 / (
        0.06629 * np.exp((V - 34.70) / 7.355)
        + 1.128 * 1e-5 * np.exp(-(V - 29.74) / 25.94)
    )
    A_xr_fast = 1 / (1 + np.exp((V + 54.81) / 38.21))
    A_xr_slow = 1 - A_xr_fast
    x_r = A_xr_fast * x_r_fast + A_xr_slow * x_r_slow
    R_Kr = 1 / ((1 + np.exp((V + 55) / 75)) * (1 + np.exp((V - 10) / 30)))
    G_Kr_bar = 0.046
    I_Kr = G_Kr_bar * np.sqrt(K_o / 5.4) * x_r * R_Kr * (V - V_K)
    PR_NaK = 0.01833
    V_Ks = (R * T / F) * np.log((K_o + PR_NaK * Na_o) / (K_i + PR_NaK * Na_i))
    x_s1_inf = 1 / (1 + np.exp(-(V + 11.60) / 8.932))
    tau_x_s1 = 817.3 + 1 / (
        2.326 * 1e-4 * np.exp((V + 48.28) / 17.80) + 0.001292 * np.exp(-(V + 210) / 230)
    )
    x_s2_inf = x_s1_inf
    tau_x_s2 = 1 / (0.01 * np.exp((V - 50) / 20) + 0.0193 * np.exp(-(V + 66.54) / 31))
    G_Ks_bar = 0.0034
    I_Ks = (
        G_Ks_bar
        * (1 + 0.6 / (1 + (3.8 * 1e-5 / Ca_i) ** (1.4)))
        * x_s1
        * x_s2
        * (V - V_Ks)
    )
    x_K1_inf = 1 / (1 + np.exp(-(V + 2.5538 * K_o + 144.59) / (1.5692 * K_o + 3.8115)))
    tau_x_K1 = 122.2 / (np.exp(-(V + 127.2) / 20.36) + np.exp((V + 236.8) / 69.33))
    R_K1 = 1 / (1 + np.exp((V + 105.8 - 2.6 * K_o) / 9.493))
    G_K1_bar = 0.1908
    I_K1 = G_K1_bar * np.sqrt(K_o) * x_K1 * R_K1 * (V - V_K)
    k_Na1 = 15
    k_Na2 = 5
    k_Na3 = 88.12
    k_asymm = 12.5
    omega_Na = 6 * 1e4
    omega_Ca = 6 * 1e4
    omega_NaCa = 5 * 1e3
    k_Ca_on = 1.5 * 1e6
    k_Ca_off = 5 * 1e3
    q_Na = 0.5224
    q_Ca = 0.1670
    h_Ca = np.exp((q_Ca * V * F) / (R * T))
    h_Na = np.exp((q_Na * V * F) / (R * T))
    K_mCaAct = 150 * 1e-6
    z_Na = 1
    z_Ca = 2
    G_NaCa_bar = 0.0008
    k_2 = k_Ca_off
    k_5 = k_Ca_off
    h_1_i = 1 + (Na_i / k_Na3) * (1 + h_Na)
    h_2_i = (Na_i * h_Na) / (k_Na3 * h_1_i)
    h_3_i = 1 / h_1_i
    h_4_i = 1 + (Na_i / k_Na1) * (1 + Na_i / k_Na2)
    h_5_i = (Na_i**2) / (h_4_i * k_Na1 * k_Na2)
    h_6_i = 1 / h_4_i
    h_7_i = 1 + (Na_o / k_Na3) * (1 + 1 / h_Na)
    h_8_i = Na_o / (k_Na3 * h_Na * h_7_i)
    h_9_i = 1 / h_7_i
    h_10_i = k_asymm + 1 + (Na_o / k_Na1) * (1 + Na_o / k_Na2)
    h_11_i = Na_o**2 / (h_10_i * k_Na1 * k_Na2)
    h_12_i = 1 / h_10_i
    k_1_i = h_12_i * Ca_o * k_Ca_on
    k_3_i_1 = h_9_i * omega_Ca
    k_3_i_2 = h_8_i * omega_NaCa
    k_3_i = k_3_i_1 + k_3_i_2
    k_4_i_1 = (h_3_i * omega_Ca) / h_Ca
    k_4_i_2 = h_2_i * omega_NaCa
    k_4_i = k_4_i_1 + k_4_i_2
    k_6_i = h_6_i * Ca_i * k_Ca_on
    k_7_i = h_5_i * h_2_i * omega_Na
    k_8_i = h_8_i * h_11_i * omega_Na
    x_1_i = k_2 * k_4_i * (k_7_i + k_6_i) + k_5 * k_7_i * (k_2 + k_3_i)
    x_2_i = k_1_i * k_7_i * (k_4_i + k_5) + k_4_i * k_6_i * (k_1_i + k_8_i)
    x_3_i = k_1_i * k_3_i * (k_7_i + k_6_i) + k_8_i * k_6_i * (k_2 + k_3_i)
    x_4_i = k_2 * k_8_i * (k_4_i + k_5) + k_3_i * k_5 * (k_1_i + k_8_i)
    E_1_i = x_1_i / (x_1_i + x_2_i + x_3_i + x_4_i)
    E_2_i = x_2_i / (x_1_i + x_2_i + x_3_i + x_4_i)
    E_3_i = x_3_i / (x_1_i + x_2_i + x_3_i + x_4_i)
    E_4_i = x_4_i / (x_1_i + x_2_i + x_3_i + x_4_i)
    allo_i = 1 / (1 + (K_mCaAct / Ca_i) ** 2)
    J_NaCa_Na_i = (
        3 * (E_4_i * k_7_i - E_1_i * k_8_i) + E_3_i * k_4_i_2 - E_2_i * k_3_i_2
    )
    J_NaCa_Ca_i = E_2_i * k_2 - E_1_i * k_1_i
    I_NaCa_i = G_NaCa_bar * 0.8 * allo_i * (z_Na * J_NaCa_Na_i + z_Ca * J_NaCa_Ca_i)
    h_1_ss = 1 + (Na_ss / k_Na3) * (1 + h_Na)
    h_2_ss = (Na_ss * h_Na) / (k_Na3 * h_1_ss)
    h_3_ss = 1 / h_1_ss
    h_4_ss = 1 + (Na_ss / k_Na1) * (1 + Na_ss / k_Na2)
    h_5_ss = (Na_ss**2) / (h_4_ss * k_Na1 * k_Na2)
    h_6_ss = 1 / h_4_ss
    h_7_ss = 1 + (Na_o / k_Na3) * (1 + 1 / h_Na)
    h_8_ss = Na_o / (k_Na3 * h_Na * h_7_ss)
    h_9_ss = 1 / h_7_ss
    h_10_ss = k_asymm + 1 + (Na_o / k_Na1) * (1 + Na_o / k_Na2)
    h_11_ss = Na_o**2 / (h_10_ss * k_Na1 * k_Na2)
    h_12_ss = 1 / h_10_ss
    k_1_ss = h_12_ss * Ca_o * k_Ca_on
    k_3_ss_1 = h_9_ss * omega_Ca
    k_3_ss_2 = h_8_ss * omega_NaCa
    k_3_ss = k_3_ss_1 + k_3_ss_2
    k_4_ss_1 = (h_3_ss * omega_Ca) / h_Ca
    k_4_ss_2 = h_2_ss * omega_NaCa
    k_4_ss = k_4_ss_1 + k_4_ss_2
    k_6_ss = h_6_ss * Ca_ss * k_Ca_on
    k_7_ss = h_5_ss * h_2_ss * omega_Na
    k_8_ss = h_8_ss * h_11_ss * omega_Na
    x_1_ss = k_2 * k_4_ss * (k_7_ss + k_6_ss) + k_5 * k_7_ss * (k_2 + k_3_ss)
    x_2_ss = k_1_ss * k_7_ss * (k_4_ss + k_5) + k_4_ss * k_6_ss * (k_1_ss + k_8_ss)
    x_3_ss = k_1_ss * k_3_ss * (k_7_ss + k_6_ss) + k_8_ss * k_6_ss * (k_2 + k_3_ss)
    x_4_ss = k_2 * k_8_ss * (k_4_ss + k_5) + k_3_ss * k_5 * (k_1_ss + k_8_ss)
    E_1_ss = x_1_ss / (x_1_ss + x_2_ss + x_3_ss + x_4_ss)
    E_2_ss = x_2_ss / (x_1_ss + x_2_ss + x_3_ss + x_4_ss)
    E_3_ss = x_3_ss / (x_1_ss + x_2_ss + x_3_ss + x_4_ss)
    E_4_ss = x_4_ss / (x_1_ss + x_2_ss + x_3_ss + x_4_ss)
    allo_ss = 1 / (1 + (K_mCaAct / Ca_ss) ** 2)
    J_NaCa_Na_ss = (
        3 * (E_4_ss * k_7_ss - E_1_ss * k_8_ss) + E_3_ss * k_4_ss_2 - E_2_ss * k_3_ss_2
    )
    J_NaCa_Ca_ss = E_2_ss * k_2 - E_1_ss * k_1_ss
    I_NaCa_ss = G_NaCa_bar * 0.2 * allo_ss * (z_Na * J_NaCa_Na_ss + z_Ca * J_NaCa_Ca_ss)
    I_NaCa = I_NaCa_i + I_NaCa_ss
    k_1_p = 949.5
    k_1_m = 182.4
    k_2_p = 687.2
    k_2_m = 39.4
    k_3_p = 1899
    k_3_m = 79300
    k_4_p = 639
    k_4_m = 40
    K_Nai_0 = 9.073
    K_Nao_0 = 27.78
    Delta = -0.1550
    K_Nai = K_Nai_0 * np.exp((Delta * V * F) / (3 * R * T))
    K_Nao = K_Nao_0 * np.exp(((1 - Delta) * V * F) / (3 * R * T))
    K_Ki = 0.5
    K_Ko = 0.3582
    MgADP = 0.05
    MgATP = 9.8
    K_MgATP = 1.698 * 1e-7
    H_p = 1e-7
    SigmaP = 4.2
    K_H_P = 1.698 * 1e-7
    K_Na_P = 224
    K_K_P = 292
    P = SigmaP / (1 + H_p / K_H_P + Na_i / K_Na_P + K_i / K_K_P)
    alpha_1 = (k_1_p * (Na_i / K_Nai) ** 3) / (
        (1 + Na_i / K_Nai) ** 3 + (1 + K_i / K_Ki) ** 2 - 1
    )
    beta_1 = k_1_m * MgADP
    alpha_2 = k_2_p
    beta_2 = (k_2_m * (Na_o / K_Nao) ** 3) / (
        (1 + Na_o / K_Nao) ** 3 + (1 + K_o / K_Ko) ** 2 - 1
    )
    alpha_3 = (k_3_p * (K_o / K_Ko) ** 2) / (
        (1 + Na_o / K_Nao) ** 3 + (1 + K_o / K_Ko) ** 2 - 1
    )
    beta_3 = (k_3_m * P * H_p) / (1 + MgATP / K_MgATP)
    alpha_4 = (k_4_p * MgATP / K_MgATP) / (1 + MgATP / K_MgATP)
    beta_4 = (k_4_m * (K_i / K_Ki) ** 2) / (
        (1 + Na_i / K_Nai) ** 3 + (1 + K_i / K_Ki) ** 2 - 1
    )
    x_1 = (
        alpha_4 * alpha_1 * alpha_2
        + beta_2 * beta_4 * beta_3
        + alpha_2 * beta_4 * beta_3
        + beta_3 * alpha_1 * alpha_2
    )
    x_2 = (
        beta_2 * beta_1 * beta_4
        + alpha_1 * alpha_2 * alpha_3
        + alpha_3 * beta_1 * beta_4
        + alpha_2 * alpha_3 * beta_4
    )
    x_3 = (
        alpha_2 * alpha_3 * alpha_4
        + beta_3 * beta_2 * beta_1
        + beta_2 * beta_1 * alpha_4
        + alpha_3 * alpha_4 * beta_1
    )
    x_4 = (
        beta_4 * beta_3 * beta_2
        + alpha_3 * alpha_4 * alpha_1
        + beta_2 * alpha_4 * alpha_1
        + beta_3 * beta_2 * alpha_1
    )
    E_1 = x_1 / (x_1 + x_2 + x_3 + x_4)
    E_2 = x_2 / (x_1 + x_2 + x_3 + x_4)
    E_3 = x_3 / (x_1 + x_2 + x_3 + x_4)
    E_4 = x_4 / (x_1 + x_2 + x_3 + x_4)
    J_NaK_Na = 3 * (E_1 * alpha_3 - E_2 * beta_3)
    J_NaK_K = 2 * (E_4 * beta_1 - E_3 * alpha_1)
    I_NaK = 30 * (z_Na * J_NaK_Na + z_K * J_NaK_K)
    np.exp_Na = np.exp(z_Na * V * F / (R * T))
    P_Nab = 3.75 * 1e-10
    I_Nab = (
        P_Nab
        * (z_Na**2)
        * ((V * (F**2)) / (R * T))
        * (Na_i * np.exp_Na - Na_o)
        / (np.exp_Na - 1)
    )
    P_Cab = 2.5 * 1e-8
    I_Cab = (
        P_Cab
        * z_Ca**2
        * ((V * F**2) / (R * T))
        * (
            (gamma_Cai * Ca_i * np.exp((z_Ca * V * F) / (R * T)) - gamma_Cao * Ca_o)
            / (np.exp((z_Ca * V * F) / (R * T)) - 1)
        )
    )
    x_Kb = 1 / (1 + np.exp(-(V - 14.48) / 18.34))
    G_Kb_bar = 0.003
    I_Kb = G_Kb_bar * x_Kb * (V - V_K)
    G_pCa_bar = 0.0005
    I_pCa = G_pCa_bar * (Ca_i / (0.0005 + Ca_i))
    tau_diff_Na = 2
    tau_diff_K = 2
    tau_diff_Ca = 0.2
    J_diff_Na = (Na_ss - Na_i) / tau_diff_Na
    J_diff_Ca = (Ca_ss - Ca_i) / tau_diff_Ca
    J_diff_K = (K_ss - K_i) / tau_diff_K
    beta_tau = 4.75
    alpha_rel = 0.5 * beta_tau
    J_rel_NP_inf = (alpha_rel * (-I_CaL)) / (1 + (1.5 / Ca_jsr) ** 8)
    tau_rel_NP = beta_tau / (1 + (0.0123 / Ca_jsr))

    if tau_rel_NP < 0.001:
        tau_rel_NP = 0.001

    beta_tau_CaMK = 1.25 * beta_tau
    alpha_rel_CaMK = 0.5 * beta_tau_CaMK
    J_rel_CaMK_inf = (alpha_rel_CaMK * (-I_CaL)) / (1 + (1.5 / Ca_jsr) ** 8)
    tau_rel_CaMK = beta_tau_CaMK / (1 + (0.0123 / Ca_jsr))

    if tau_rel_NP < 0.001:
        tau_rel_CaMK = 0.001

    Phi_rel_CaMK = 1 / (1 + K_m_CaMK / CaMK_active)
    J_rel = (1 - Phi_rel_CaMK) * J_rel_NP + Phi_rel_CaMK * J_rel_CaMK
    J_up_NP = (0.004375 * Ca_i) / (0.00092 + Ca_i)
    DeltaK_m_PLB_bar = 0.00017
    DeltaJ_up_CaMK_bar = 1.75
    J_up_CaMK = (1 + DeltaJ_up_CaMK_bar) * (
        (0.004375 * Ca_i) / (0.00092 - DeltaK_m_PLB_bar + Ca_i)
    )
    Phi_up_CaMK = 1 / (1 + K_m_CaMK / CaMK_active)
    J_leak = (0.0039375 * Ca_nsr) / 15
    J_up = (1 - Phi_up_CaMK) * J_up_NP + Phi_up_CaMK * J_up_CaMK - J_leak
    tau_tr = 100
    J_tr = (Ca_nsr - Ca_jsr) / tau_tr
    CMDN_bar = 0.05
    K_m_CMDN = 0.00238
    TRPN_bar = 0.07
    K_m_TRPN = 0.0005
    BSR_bar = 0.047
    K_m_BSR = 0.00087
    BSL_bar = 1.124
    K_m_BSL = 0.0087
    CSQN_bar = 10
    K_m_CSQN = 0.8
    beta_Cai = 1 / (
        1
        + (CMDN_bar * K_m_CMDN) / (K_m_CMDN + Ca_i) ** 2
        + (TRPN_bar * K_m_TRPN) / (K_m_TRPN + Ca_i) ** 2
    )
    beta_Cass = 1 / (
        1
        + (BSR_bar * K_m_BSR) / (K_m_BSR + Ca_ss) ** 2
        + (BSL_bar * K_m_BSL) / (K_m_BSL + Ca_ss) ** 2
    )
    beta_Cajsr = 1 / (1 + (CSQN_bar * K_m_CSQN) / (K_m_CSQN + Ca_jsr) ** 2)

    # Initialize derivatives array
    dydt = np.zeros_like(y)

    # Current summation terms for voltage equation
    I_sum = (
        I_Na_fast
        + I_Na_late
        + I_to
        + I_CaL
        + I_CaNa
        + I_CaK
        + I_Kr
        + I_Ks
        + I_K1
        + I_NaCa
        + I_NaK
        + I_Nab
        + I_Cab
        + I_Kb
        + I_pCa
    )

    # Voltage and ionic concentrations
    if t <= T_stim:
        dydt[0] = -1 / C_m * I_sum + I_app / C_m
        dydt[3] = (
            -(I_to + I_Kr + I_Ks + I_K1 + I_Kb - I_app - 2 * I_NaK)
            * (A_cap / (F * v_myo))
            + J_diff_K * v_ss / v_myo
        )
    else:
        dydt[0] = -1 / C_m * I_sum
        dydt[3] = (
            -(I_to + I_Kr + I_Ks + I_K1 + I_Kb - 2 * I_NaK) * (A_cap / (F * v_myo))
            + J_diff_K * v_ss / v_myo
        )

    # Sodium and calcium dynamics
    dydt[1] = (
        -(I_Na_fast + I_Na_late + 3 * I_NaCa_i + 3 * I_NaK + I_Nab)
        * (A_cap / (F * v_myo))
        + J_diff_Na * v_ss / v_myo
    )
    dydt[2] = -(I_CaNa + 3 * I_NaCa_ss) * (A_cap / (F * v_ss)) - J_diff_Na
    dydt[4] = -I_CaK * (A_cap / (F * v_ss)) - J_diff_K

    # Calcium handling
    dydt[5] = beta_Cai * (
        -(I_pCa + I_Cab - 2 * I_NaCa_i) * A_cap / (2 * F * v_myo)
        - J_up * v_nsr / v_myo
        + J_diff_Ca * v_ss / v_myo
    )
    dydt[6] = beta_Cass * (
        -(I_CaL - 2 * I_NaCa_ss) * A_cap / (2 * F * v_ss)
        + J_rel * v_jsr / v_ss
        - J_diff_Ca
    )
    dydt[7] = J_up - J_tr * v_jsr / v_nsr
    dydt[8] = beta_Cajsr * (J_tr - J_rel)

    # Gating variables
    dydt[9] = (m_inf - y[9]) / tau_m
    dydt[10] = (h_inf - y[10]) / tau_h_fast
    dydt[11] = (h_inf - y[11]) / tau_h_slow
    dydt[12] = (j_inf - y[12]) / tau_j
    dydt[13] = (h_CaMK_inf - y[13]) / tau_h_CaMK_slow
    dydt[14] = (j_CaMK_inf - y[14]) / tau_j_CaMK
    dydt[15] = (m_L_inf - y[15]) / tau_m_L
    dydt[16] = (h_L_inf - y[16]) / tau_h_L
    dydt[17] = (h_L_CaMK_inf - y[17]) / tau_h_L_CaMK
    dydt[18] = (alpha_inf - y[18]) / tau_a
    dydt[19] = (i_inf - y[19]) / tau_i_fast
    dydt[20] = (i_inf - y[20]) / tau_i_slow
    dydt[21] = (a_CaMK_inf - y[21]) / tau_a_CaMK
    dydt[22] = (i_CaMK_inf - y[22]) / tau_i_CaMK_fast
    dydt[23] = (i_CaMK_inf - y[23]) / tau_i_CaMK_slow
    dydt[24] = (d_inf - y[24]) / tau_d
    dydt[25] = (f_inf - y[25]) / tau_f_fast
    dydt[26] = (f_inf - y[26]) / tau_f_slow
    dydt[27] = (f_Ca_inf - y[27]) / tau_f_Ca_fast
    dydt[28] = (f_Ca_inf - y[28]) / tau_f_Ca_slow
    dydt[29] = (j_Ca_inf - y[29]) / tau_j_Ca
    dydt[30] = alpha_n * k_p2_n - y[30] * k_m2_n
    dydt[31] = (f_CaMK_inf - y[31]) / tau_f_CaMK_fast
    dydt[32] = (f_Ca_CaMK_inf - y[32]) / tau_f_Ca_CaMK_fast
    dydt[33] = (x_r_inf - y[33]) / tau_xr_fast
    dydt[34] = (x_r_inf - y[34]) / tau_xr_slow
    dydt[35] = (x_s1_inf - y[35]) / tau_x_s1
    dydt[36] = (x_s2_inf - y[36]) / tau_x_s2
    dydt[37] = (x_K1_inf - y[37]) / tau_x_K1
    dydt[38] = (J_rel_NP_inf - y[38]) / tau_rel_NP
    dydt[39] = (J_rel_CaMK_inf - y[39]) / tau_rel_CaMK
    dydt[40] = alpha_CaMK * CaMK_bound * (CaMK_bound + y[40]) - beta_CaMK * y[40]

    return dydt
