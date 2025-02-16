import streamlit as st
import sys

sys.path.append("..")
from utilities import plot_input, plot_outputs, plot_errors


def fhn_page():
    st.title("FitzHugh-Nagumo model", anchor=False)

    st.markdown(
        """
        The FitzHugh-Nagumo model is a simplified model of the Hodgkin-Huxley model, which describes
        the dynamics of action potentials in neurons. The model consists of two ordinary differential
        equations that describe the membrane potential and the recovery variable.
    """
    )

    ## Selection of the model
    str_model = st.sidebar.selectbox(
        "Select the model",
        ["best", "best_samedofs"],
    )

    ## Selection of the indexes
    sample_idxs = []
    initial_values = [250, 280, 310, 340]
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
                    plot_input("FHN", str_model, "input", sample_idx, "Current (mA)"),
                    key=f"input_{sample_idx}",
                )
            else:
                st.plotly_chart(
                    plot_input("FHN", str_model, "input", sample_idx),
                    key=f"input_{sample_idx}",
                )

    ## Plot of the output and the errors
    # Store the number of sets in session state
    if "num_variables_fhn" not in st.session_state:
        st.session_state.num_variables_fhn = 1

    # Display all the variables
    for var in range(st.session_state.num_variables_fhn):

        # Selection of the variable
        cols = st.columns(4)

        with cols[0]:
            str_variable = st.selectbox(
                "Select the variable to plot",
                ["V", "w"],
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
                            "FHN", str_model, str_variable, sample_idx, "Voltage (mV)"
                        ),
                        key=f"output_{sample_idx}_var_{var}",
                    )
                else:
                    st.plotly_chart(
                        plot_outputs("FHN", str_model, str_variable, sample_idx),
                        key=f"output_{sample_idx}_var_{var}",
                    )
        # errors
        cols = st.columns(4)
        for idx, (col, sample_idx) in enumerate(zip(cols, sample_idxs)):
            with col:
                if idx == 0:
                    st.plotly_chart(
                        plot_errors(
                            "FHN",
                            str_model,
                            str_variable,
                            sample_idx,
                            "Module of the error",
                        ),
                        key=f"error_{sample_idx}_var_{var}",
                    )
                else:
                    st.plotly_chart(
                        plot_errors("FHN", str_model, str_variable, sample_idx),
                        key=f"error_{sample_idx}_var_{var}",
                    )

    # Button to add more sets
    if st.button("Add another variable to plot"):
        st.session_state.num_variables_fhn += 1


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
    fhn_page()
