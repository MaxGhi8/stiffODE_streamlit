import streamlit as st


# Main page content
def main():

    # Title
    st.title(
        "Learning Ionic Model Dynamics Using Fourier Neural Operators", anchor=False
    )
    # Create a container with custom CSS
    st.markdown(
        """
        <style>
        .profile-name {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 15px;
            font-size: 1.2em;
            font-weight: bold;
            color: var(--text-color);
        }
        .profile-title {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 5px;
            font-size: 0.9em;
            color: var(--text-color);
        }
        .profile-title-secondary {
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            top: -15px;
            margin-bottom: -10px;
            font-size: 0.9em;
            color: var(--text-color);
        }
        .email-button {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 10px;
        }
        .email-button a {
            background-color: #0066cc;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 0.9em;
            transition: background-color 0.3s;
        }
        .email-button a:hover {
            background-color: #0052a3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create columns for the cards
    col1, col2, col3 = st.columns([1, 1, 1], border=True)

    with col1:
        with st.container():

            cols = st.columns([1, 5, 1])
            with cols[1]:
                st.image("luca_pellegrini.png", width=120, use_container_width=False)

            st.markdown(
                "<p class='profile-name'>Luca Pellegrini</p>", unsafe_allow_html=True
            )
            st.markdown(
                "<p class='profile-title'>University of Pavia</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='profile-title-secondary'>Euler Institute Switzerland</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="email-button">
                    <a href="mailto:luca.pellegrini02@universitadipavia.it">
                        <span class="email-emoji">🖂</span> Contact Me
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        with st.container():

            cols = st.columns([1, 5, 1])
            with cols[1]:
                st.image(
                    "massimiliano_ghiotto.jpg", width=120, use_container_width=False
                )

            st.markdown(
                "<p class='profile-name'>Massimiliano Ghiotto</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='profile-title'>University of Pavia</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="email-button">
                    <a href="mailto:massimiliano.ghiotto01@universitadipavia.it">
                        <span class="email-emoji">🖂</span> Contact Me
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col3:
        with st.container():

            cols = st.columns([1, 5, 1])
            with cols[1]:
                st.image(
                    "luca_franco_pavarino.png", width=120, use_container_width=False
                )

            st.markdown(
                "<p class='profile-name'>Luca Franco Pavarino</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='profile-title'>University of Pavia</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="email-button">
                    <a href="mailto:luca.pavarino@unipv.it">
                        <span class="email-emoji">🖂</span> Contact Me
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Description
    st.header("Project Description", anchor=False)
    st.markdown(
        """
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
    """
    )

    # FNO video
    st.header("One dimensional Fourier Neural Operator visualization", anchor=False)
    st.markdown(
        """
        In this video, we show the architecture of the Fourier Neural Operator for the one-dimensional case, that is the case of interest for our work.
    """
    )
    st.video("FNO_architecture_1d.mp4", loop=True, autoplay=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="FNO ionic models",
        page_icon=":moyai:",
        menu_items={
            "Report a bug": "https://github.com/MaxGhi8/stiffODE_streamlit/issues",
            # "About": "sium",
        },
    )
    main()
