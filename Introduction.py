import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Main page configuration
st.set_page_config(page_title="ODE Models Visualization")

# Main page content
def main():

    # Title
    st.title("Learning Ionic Model Dynamics Using Fourier Neural Operators")

    # Define the CSS with better styling
    css = """
    <style>
    .author-grid {
        display: grid;
        gap: 20px;
        padding: 20px 0;
    }

    .author-card {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid white;
    }

    .author-card:hover {
        transform: translateY(-5px);
    }

    .author-image {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        margin: 0 auto 15px auto;
        border: 3px solid white;
        padding: 3px;
    }

    .author-name {
        color: var(--text-color);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .author-affiliation {
        color: var(--text-color);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    </style>
    """

    # Define the HTML template for the card
    card_html = """
    <div class="author-card">
        <img class="author-image" src="file://{}">
        <div class="author-name">{}</div>
        <div class="author-affiliation">{}</div>
    </div>
    """

    # Apply the CSS
    st.markdown(css, unsafe_allow_html=True)

    # Create a container for better spacing
    with st.container():
        # Create columns for the cards
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown(card_html.format(
                "massimiliano_ghiotto.jpg",
                "Massimiliano Ghiotto",
                "University of Pavia"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(card_html.format(
                "luca_pellegrini.jpeg",
                "Luca Franco Pavarino",
                "University of Pavia"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(card_html.format(
                "luca_pellegrini.jpeg",
                "Luca Pellegrini",
                "University of Pavia,<br>Euler Institute Switzerland"
            ), unsafe_allow_html=True)



    # Description    
    st.header("Project Description")
    st.markdown("""
        This web application serves as an interactive visualization tool for comparing and analyzing different Ordinary Differential Equation (ODE) models commonly used in cardiac electrophysiology.
        The featured models include the Hodgkin-Huxley (HH) model, which describes action potential generation in neurons, 
        the FitzHugh-Nagumno (FHN) model that is a simplification with two variables of the original HH model, 
        and the O'Hara-Rudy (ORD) model, which simulates with 41 variables the human ventricular cell behavior.

        For each mathematical model, the application provides comprehensive visualizations across multiple pages. Users can examine:

        - Input functions that drive the model simulations.
        - True solutions derived from analytical methods or high-precision numerical calculations, compared with approximate solutions with Fourier Neural Operator (FNO) models.
        - Error analysis showing the absolute value of the difference between true and approximate solutions, helping users understand the accuracy and limitations of different neural operator models.

        These visualizations help researchers, students, and practitioners better understand the behavior of these complex biological systems and evaluate the effectiveness of different neural operator models.
        Moreover, for the OHara-Rudy model, we have 41 different and coupled variables with different behaviors, which makes very complex to analyze the behavior all in once inside an article. 
        For these reasons, we have created this web application to help the reader to understand the behavior of the model in a more interactive and comprehensive way.
    """)
    
    # FNO video
    st.header("One dimensional Fourier Neural Operator visualization")
    st.markdown("""
        In this video, we show the architecture of the Fourier Neural Operator for the one-dimensional case, that is the case of interest for our work.
    """)
    st.video("FNO_architecture_2d.mp4", loop=True, autoplay=True)


if __name__ == "__main__":
    main()