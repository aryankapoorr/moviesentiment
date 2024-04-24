import streamlit as st
from movie_scores import movie_scores_page
from test_model import test_model_page
from model_training_results import model_training_results_page
from about import about_page

def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded",
                   page_title="Movie Review Sentiment Analysis Scoring", page_icon="🎥")
    st.title("Movie Review Sentiment Analysis")
    st.sidebar.title("Navigation")
    selected_tab = st.sidebar.radio(
        "Go to", ["Movie Scores", "Test the model!", "Model Training Results", "About"], index=0)

    if selected_tab == "Movie Scores":
        movie_scores_page()

    elif selected_tab == "Test the model!":
        test_model_page()

    elif selected_tab == "Model Training Results":
        model_training_results_page()

    elif selected_tab == "About":
        about_page()

if __name__ == "__main__":
    main()
