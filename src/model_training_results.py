import streamlit as st

def model_training_results_page():
    st.write("Each data point represents a review that was tested to capture performance after the model was trained.")
    st.write('<iframe title="Sentiment Analysis Test Results" aria-label="Scatter Plot" id="datawrapper-chart-2INNM" src="https://datawrapper.dwcdn.net/2INNM/5/" scrolling="no" frameborder="0" style="width: 80%; min-width: 100% !important; border: none;" height="682" data-external="1"></iframe>', unsafe_allow_html=True)

    st.write("\n\nAfter observing the distribution of data points in the smaller test size, it become clear that there was no obvious boundary to discern between positive and negative reviews. Therefore, generating a score for a review was done with the decimal score, instead of a binary positive/negative.")
    st.write('<iframe title="Sentiment Analysis Test Results (Shortened)" aria-label="Scatter Plot" id="datawrapper-chart-P5oxs" src="https://datawrapper.dwcdn.net/P5oxs/3/" scrolling="no" frameborder="0" style="width: 80%; min-width: 100% !important; border: none;" height="682", unsafe_allow_html=True')
