import sys

import streamlit as st

sys.path.append("..")
from utilities import plot_errors, plot_input, plot_outputs


def ord_page():
    st.title("OHara-Rudy model", anchor=False)

    st.markdown(
        """
        The O'Hara-Rudy model is a mathematical model that describes the generation and propagation
        of action potentials in cardiomyocytes. The model consists of 42 ordinary differential equations that
        describe the dynamics of the membrane potential and the gating variables of ion channels.
    """
    )

    ## Selection of the model
    str_model = st.sidebar.selectbox(
        "Select the model",
        ["best", "best_samedofs"],
    )

    ## Selection of the indexes
    sample_idxs = []
    initial_values = [270, 310, 330, 360]
    for idx, initial_value in enumerate(initial_values):
        sample_idx = st.sidebar.number_input(
            f"Index of the {idx + 1} column",
            min_value=0,
            max_value=374,
            value=initial_value,
            step=1,
            key=f"sample_idx_{idx}",
        )
        sample_idxs.append(sample_idx)

    ## Plot of the input
    # st.header("Input Function")
    cols = st.columns(4)
    for idx, (col, sample_idx) in enumerate(zip(cols, sample_idxs)):
        with col:
            if idx == 0:
                st.plotly_chart(
                    plot_input("ORD", str_model, "input", sample_idx, "Current (mA)"),
                    key=f"input_{idx}_{sample_idx}",
                )
            else:
                st.plotly_chart(
                    plot_input("ORD", str_model, "input", sample_idx),
                    key=f"input_{idx}_{sample_idx}",
                )

    ## Plot of the output and the errors
    # Store the number of sets in session state
    if "num_variables_ord" not in st.session_state:
        st.session_state.num_variables_ord = 1

    # Display all the variables
    for var in range(st.session_state.num_variables_ord):

        # Selection of the variable
        cols = st.columns(4)

        with cols[0]:
            str_variable = st.selectbox(
                "Select the variable to plot",
                [
                    "CaMK_trap_dataset",
                    "Ca_i_dataset",
                    "Ca_jsr_dataset",
                    "Ca_nsr_dataset",
                    "Ca_ss_dataset",
                    "J_rel_CaMK_dataset",
                    "J_rel_NP_dataset",
                    "K_i_dataset",
                    "K_ss_dataset",
                    "Na_i_dataset",
                    "Na_ss_dataset",
                    "V_dataset",
                    "a_CaMK_dataset",
                    "a_dataset",
                    "d_dataset",
                    "f_CaMK_fast_dataset",
                    "f_Ca_CaMK_fast_dataset",
                    "f_Ca_fast_dataset",
                    "f_Ca_slow_dataset",
                    "f_fast_dataset",
                    "f_slow_dataset",
                    "h_CaMK_slow_dataset",
                    "h_L_CaMK_dataset",
                    "h_L_dataset",
                    "h_fast_dataset",
                    "h_slow_dataset",
                    "i_CaMK_fast_dataset",
                    "i_CaMK_slow_dataset",
                    "i_fast_dataset",
                    "i_slow_dataset",
                    "j_CaMK_dataset",
                    "j_Ca_dataset",
                    "j_dataset",
                    "m_L_dataset",
                    "m_dataset",
                    "n_dataset",
                    "x_k1_dataset",
                    "x_r_fast_dataset",
                    "x_r_slow_dataset",
                    "x_s1_dataset",
                    "x_s2_dataset",
                ],
                index=var,
                key=f"variable_{var}",  # Unique key for each selectbox
            )

        # outputs
        cols = st.columns(4)
        for idx, (col, sample_idx) in enumerate(zip(cols, sample_idxs)):
            with col:
                if idx == 0:
                    st.plotly_chart(
                        plot_outputs(
                            "ORD",
                            str_model,
                            str_variable,
                            sample_idx,
                            f"Output {str_variable.replace('_', ' ')[:-7]}",
                        ),
                        key=f"output_{idx}_{sample_idx}_var_{var}",
                    )
                else:
                    st.plotly_chart(
                        plot_outputs("ORD", str_model, str_variable, sample_idx),
                        key=f"output_{idx}_{sample_idx}_var_{var}",
                    )
        # errors
        cols = st.columns(4)
        for idx, (col, sample_idx) in enumerate(zip(cols, sample_idxs)):
            with col:
                if idx == 0:
                    st.plotly_chart(
                        plot_errors(
                            "ORD",
                            str_model,
                            str_variable,
                            sample_idx,
                            "Module of the error",
                        ),
                        key=f"error_{idx}_{sample_idx}_var_{var}",
                    )
                else:
                    st.plotly_chart(
                        plot_errors("ORD", str_model, str_variable, sample_idx),
                        key=f"error_{idx}_{sample_idx}_var_{var}",
                    )

    # Button to add more sets
    cols = st.columns(4)
    with cols[0]:
        if st.session_state.num_variables_ord < 41:
            if st.button("Add another variable to plot", key="add_var"):
                st.session_state.num_variables_ord += 1
                st.rerun()
        else:
            st.markdown("You have plotted all the variables for this problem.")
    with cols[1]:
        if st.session_state.num_variables_ord > 1:
            if st.button("Remove a variable to plot", key="remove_var"):
                st.session_state.num_variables_ord -= 1
                st.rerun()


if __name__ == "__main__":
    st.set_page_config(
        page_title="FNO ionic models",
        layout="wide",
        page_icon=":anatomical_heart:",
        menu_items={
            "Report a bug": "https://github.com/MaxGhi8/stiffODE_streamlit/issues",
            # "About": "sium",
        },
    )
    ord_page()
