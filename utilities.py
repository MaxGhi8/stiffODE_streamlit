import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy.io import loadmat


@st.cache_resource
def read_data(problem: str, str_model: str):

    match problem:
        case "FHN":
            data = loadmat(f"data/fhn_trainL2_n_375_points_1260_tf_100_{str_model}.mat")
        case "HH":
            data = loadmat(f"data/hh_trainL2_n_375_points_1260_tf_100_{str_model}.mat")
        case "ORD":
            data = loadmat(
                f"data/ord_trainL2_n_375_points_3360_tf_1000_{str_model}.mat"
            )
        case _:
            raise ValueError(f"Invalid problem: {problem}")

    return data


@st.cache_data
def plot_input(
    problem: str, str_model: str, attribute: str, sample_idx: int, ylabel: str = None
):
    """
    problem: str
        The name of the problem selected by the user.

    str_model: str
        The name of the model selected by the user (best, best_samedofs).

    attribute: str
        The attribute of the model to be plotted.

    sample_idx: int
        The index of the sample to be plotted.

    index_plot: int
        The index of the plot to be displayed.

    ylabel: str
        The label of the y-axis.
    """

    data = read_data(problem, str_model)

    y_data = data[attribute][sample_idx]
    if attribute == "input":
        y_data = np.round(y_data, 4)

    x_data = np.linspace(0, 100, len(y_data))

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
        # plot_bgcolor="#f5f7fa",
        # paper_bgcolor="#ffffff",
        width=400,
        height=300,
        margin=dict(
            l=20,  # left margin
            r=20,  # right margin
            t=20,  # top margin
            b=20,  # bottom margin
        ),
    )

    return fig


@st.cache_data
def plot_outputs(
    problem: str, str_model: str, attribute: str, sample_idx: int, ylabel: str = None
):
    """
    Same as plot_input, but with the plot of two samples overlapped.
    """

    data = read_data(problem, str_model)

    y_data_exact = data[f"{attribute}_exact"][sample_idx]
    y_data_appro = data[f"{attribute}_pred"][sample_idx]
    x_data = np.linspace(0, 100, len(y_data_exact))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data_appro,
            mode="lines",
            line=dict(color="red"),
            name="FNO",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data_exact,
            mode="lines",
            line=dict(color="blue"),
            name="Exact",
        )
    )

    if ylabel:
        fig.update_layout(
            yaxis_title=ylabel,
        )

    fig.update_layout(
        xaxis_title="Time (ms)",
        font=dict(family="Arial", size=12),
        # plot_bgcolor="#f5f7fa",
        # paper_bgcolor="#ffffff",
        width=400,
        height=300,
        margin=dict(
            l=20,  # left margin
            r=20,  # right margin
            t=20,  # top margin
            b=20,  # bottom margin
        ),
        legend=dict(x=0.5, y=1.0, xanchor="center", yanchor="bottom", orientation="h"),
    )

    return fig


@st.cache_data
def plot_errors(
    problem: str, str_model: str, attribute: str, sample_idx: int, ylabel: str = None
):
    """
    Same as plot_input, but with the plot of the errors between the exact solution and the approximation.
    """

    data = read_data(problem, str_model)

    y_data_exact = data[f"{attribute}_exact"][sample_idx]
    y_data_appro = data[f"{attribute}_pred"][sample_idx]
    x_data = np.linspace(0, 100, len(y_data_exact))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=np.abs(y_data_appro - y_data_exact),
            mode="lines",
            name="error",
        )
    )

    if ylabel:
        fig.update_layout(
            yaxis_title=ylabel,
        )

    fig.update_yaxes(type="log")

    fig.update_layout(
        xaxis_title="Time (ms)",
        font=dict(family="Arial", size=12),
        # plot_bgcolor="#f5f7fa",
        # paper_bgcolor="#ffffff",
        width=400,
        height=300,
        margin=dict(
            l=20,  # left margin
            r=20,  # right margin
            t=20,  # top margin
            b=20,  # bottom margin
        ),
    )

    return fig
