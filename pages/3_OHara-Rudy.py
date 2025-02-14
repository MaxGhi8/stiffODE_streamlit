import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")


def fhn_page():
    st.title("OHara-Rudy model")

    st.markdown(
        """
        The O'Hara-Rudy model is a mathematical model that describes the generation and propagation
        of action potentials in cardiomyocytes. The model consists of 49 ordinary differential equations that
        describe the dynamics of the membrane potential and the gating variables of ion channels.
    """
    )

    st.header("Input Function")

    # Create columns for visualization
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        plot_function()

    with col2:
        plot_true_solution()

    with col3:
        plot_approximate_solution()

    with col4:
        plot_error()


def placeholder_plot():
    # Placeholder function for actual plots
    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    return fig


def plot_function():
    # Plot input function (replace with actual function)
    st.pyplot(placeholder_plot())


def plot_true_solution():
    # Plot true solution (replace with actual solution)
    st.pyplot(placeholder_plot())


def plot_approximate_solution():
    # Plot approximate solution (replace with actual approximation)
    st.pyplot(placeholder_plot())


def plot_error():
    # Plot error (replace with actual error)
    st.pyplot(placeholder_plot())


if __name__ == "__main__":
    fhn_page()
