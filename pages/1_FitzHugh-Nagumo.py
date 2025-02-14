import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")


def placeholder_plot(index: int):
    # Example data (replace with actual data)
    x = [0, 1]
    y = [0, 1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
        )
    )

    if index == 0:
        fig.update_layout(
            yaxis_title="Voltage (mV)",
        )

    fig.update_layout(
        xaxis_title="Time (ms)",
        font=dict(family="Arial", size=12),
        # plot_bgcolor="#f5f7fa",
        # paper_bgcolor="#ffffff",
    )

    return fig


def fhn_page():
    st.title("FitzHugh-Nagumo model")

    st.markdown(
        """
        The FitzHugh-Nagumo model is a simplified model of the Hodgkin-Huxley model, which describes
        the dynamics of action potentials in neurons. The model consists of two ordinary differential
        equations that describe the membrane potential and the recovery variable.
    """
    )

    st.header("Input Function")

    # Create columns for visualization
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.plotly_chart(placeholder_plot(0), key="input_1")

    with col2:
        st.plotly_chart(placeholder_plot(1), key="input_2")

    with col3:
        st.plotly_chart(placeholder_plot(2), key="input_3")

    with col4:
        st.plotly_chart(placeholder_plot(3), key="input_4")


if __name__ == "__main__":
    fhn_page()
