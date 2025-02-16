import streamlit as st
import plotly.graph_objects as go
import torch
import numpy as np
import sys

sys.path.append("..")
from models.FNO import FNO

device = torch.device("cpu")


@st.cache_data
def create_input_current(
    n_points: int, duration: float, amplitude: float, max_duration: float
):
    time_vector = torch.linspace(0, max_duration, n_points).to(device)
    mask = time_vector <= duration
    input_current = torch.zeros_like(time_vector)
    input_current[mask] = amplitude
    return input_current


@st.cache_resource
def load_model(str_problem: str):

    # FHN trained model
    if str_problem == "FitzHugh-Nagumo":
        model = FNO(
            problem_dim=1,
            in_dim=1,
            d_v=64,
            out_dim=2,
            L=5,
            modes=12,
            fun_act="gelu",
            weights_norm="Kaiming",
            arc="Zongyi",
            RNN=False,
            FFTnorm=None,
            padding=5,
            device=device,
            retrain_fno=4,
        )
        model.load_state_dict(
            torch.load(
                "models/model_FNO_1D_FitzHughNagumo_best_samedofs_state_dict",
                weights_only=False,
                map_location=device,
            )
        )

    # HH trained model
    elif str_problem == "Hodgkin-Huxley":
        model = torch.load(
            "models/model_FNO_1D_HodgkinHuxley_best_samedofs_state_dict",
            weights_only=False,
            map_location=device,
        )

    # ORD trained model
    elif str_problem == "O'Hara-Rudy":
        pass  # todo

    else:
        raise ValueError(f"Invalid problem: {str_problem}")

    return model


def plot_tensor(tensor, str_problem: str, ylabel: str = None):
    """
    tensor: torch.Tensor
        The tensor to be plotted.

    str_problem: str
        The name of the problem selected by the user.

    ylabel: str
        The label of the y-axis.
    """
    y_data = tensor.cpu().numpy()
    y_data = np.round(y_data, 4)

    if str_problem == "FitzHugh-Nagumo" or str_problem == "Hodgkin-Huxley":
        x_data = np.linspace(0, 100, len(y_data))
    else:
        x_data = np.linspace(0, 1000, len(y_data))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="lines",
        )
    )

    if ylabel:
        fig.update_layout(
            yaxis_title=ylabel,
        )

    fig.update_layout(
        xaxis_title="Time (ms)",
        font=dict(family="Arial", size=12),
        width=400,
        height=300,
        margin=dict(
            l=20,  # left margin
            r=20,  # right margin
            t=30,  # top margin
            b=20,  # bottom margin
        ),
    )

    return fig


def test_model_page():
    st.title("Test models with user-defined inputs", anchor=False)

    st.markdown(
        """This page allows you to test the models with user-defined inputs.
            The user can choose the problem (FitzHugh-Nagumo, Hodgkin-Huxley, or O'Hara-Rudy)
            and define the input current of his choice (defining amplitude of the applied current and the duration of the stimulus).
            Then we plot the input, the high fidelity solution computed with standard numerical analysis techniques,
            and the approximated solution computed with our trained Fourier Neural Operator model and then the absolute value of the error computed pointwise.
            Here we use our lightweight trained model (with approximated 500k trainable parameters) for the sake of speed, see the paper for more information."""
    )

    ## Selection of problem and input by the user
    cols = st.columns(3)

    with cols[0]:
        # Selection of the model
        str_problem = st.selectbox(
            "Select the problem",
            ["FitzHugh-Nagumo", "Hodgkin-Huxley", "O'Hara-Rudy"],
        )

    with cols[1]:
        # Amplitude current
        # todo: trovare max e minimo in base al problema
        amplitude = st.number_input(
            "Select the amplitude of the stimulus (mA)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.01,
        )

    with cols[2]:
        # Duration current
        if str_problem == "FitzHugh-Nagumo" or str_problem == "Hodgkin-Huxley":
            max_duration = 100.0
            duration = st.number_input(
                "Select the duration of the stimulus (ms)",
                min_value=0.0,
                max_value=max_duration,
                value=10.0,
                step=0.01,
            )
            n_points = 1260
        else:
            max_duration = 1000.0
            duration = st.number_input(
                "Select the duration of the stimulus (ms)",
                min_value=0.0,
                max_value=max_duration,
                value=1.0,
                step=0.01,
            )
            n_points = 3360

    ## FNO evaluation
    # Create the input tensor
    input_tensor = create_input_current(n_points, duration, amplitude, max_duration)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(-1)
    # todo manca scrivere varianza e media di input e output

    # Load the model
    model = load_model(str_problem)
    model.eval()

    # Compute the output
    with torch.no_grad():
        output_tensor = model(input_tensor)

    ## Plot results
    cols_ = st.columns(3)

    with cols_[0]:
        # Plot the input
        st.plotly_chart(
            plot_tensor(input_tensor.squeeze(), str_problem, "Current (mA)"),
            key=f"input",
        )

    with cols_[1]:
        # Plot the input
        st.plotly_chart(
            plot_tensor(output_tensor[:, :, 0].squeeze(), str_problem, "Current (mA)"),
            key=f"output",
        )


if __name__ == "__main__":
    st.set_page_config(
        page_title="FNO ionic models",
        layout="wide",
        page_icon=":moyai:",
        menu_items={
            "Report a bug": "https://github.com/MaxGhi8/stiffODE_streamlit/issues",
            # "About": "sium",
        },
    )
    test_model_page()
