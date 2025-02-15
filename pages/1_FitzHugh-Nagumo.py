import streamlit as st
import sys

sys.path.append("..")
from utilities import plot_input, plot_outputs, plot_errors


def fhn_page():
    st.title("FitzHugh-Nagumo model")

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
    sample_idx_0 = st.sidebar.number_input(
        "Index of the first column",
        min_value=0,
        max_value=374,
        value=250,  # default value
        step=1,  # increment by 1
    )
    sample_idx_1 = st.sidebar.number_input(
        "Index of the second column",
        min_value=0,
        max_value=374,
        value=280,
        step=1,
    )
    sample_idx_2 = st.sidebar.number_input(
        "Index of the third column",
        min_value=0,
        max_value=374,
        value=310,
        step=1,
    )
    sample_idx_3 = st.sidebar.number_input(
        "Index of the fourth column",
        min_value=0,
        max_value=374,
        value=340,
        step=1,
    )

    sample_idxs = (sample_idx_0, sample_idx_1, sample_idx_2, sample_idx_3)

    ## Plot of the input
    # st.header("Input Function")
    cols = st.columns(4)
    for idx, (col, sample_idx) in enumerate(zip(cols, sample_idxs)):
        with col:
            if idx == 0:
                st.plotly_chart(
                    plot_input(str_model, "input", sample_idx, "Current (mA)"),
                    key=f"input_{sample_idx}",
                )
            else:
                st.plotly_chart(
                    plot_input(str_model, "input", sample_idx),
                    key=f"input_{sample_idx}",
                )

    ## Plot of the output and the errors
    # Store the number of sets in session state
    if "num_variables" not in st.session_state:
        st.session_state.num_variables = 1

    # Display all the variables
    for var in range(st.session_state.num_variables):

        # Selection of the variable
        str_variable = st.selectbox(
            "Select the variable to plot",
            ["V", "w"],
            key=f"variable_{var}",  # Unique key for each selectbox
        )

        # outputs
        cols = st.columns(4)
        for idx, (col, sample_idx) in enumerate(zip(cols, sample_idxs)):
            with col:
                if idx == 0:
                    st.plotly_chart(
                        plot_outputs(
                            str_model, str_variable, sample_idx, "Voltage (mV)"
                        ),
                        key=f"output_{sample_idx}_var_{var}",
                    )
                else:
                    st.plotly_chart(
                        plot_outputs(str_model, str_variable, sample_idx),
                        key=f"output_{sample_idx}_var_{var}",
                    )
        # errors
        cols = st.columns(4)
        for idx, (col, sample_idx) in enumerate(zip(cols, sample_idxs)):
            with col:
                if idx == 0:
                    st.plotly_chart(
                        plot_errors(
                            str_model, str_variable, sample_idx, "Module of the error"
                        ),
                        key=f"error_{sample_idx}_var_{var}",
                    )
                else:
                    st.plotly_chart(
                        plot_errors(str_model, str_variable, sample_idx),
                        key=f"error_{sample_idx}_var_{var}",
                    )

    # Button to add more sets
    if st.button("Add another variable to plot"):
        st.session_state.num_variables += 1


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    fhn_page()
