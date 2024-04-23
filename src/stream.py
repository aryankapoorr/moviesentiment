import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Set page configuration
st.set_page_config(layout="wide", initial_sidebar_state="expanded",
                   page_title="NLP Sentiment Analysis Project")

# Load the CSV file


@st.cache_data
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data


# Assuming you have a CSV with movie data and poster URLs
data = load_data('Sentiment Analysis Data - Sheet1.csv')

# Load the Keras model


def load_model():
    # Assuming the model file is 'sentiment_analysis_model.h5'
    model = tf.keras.models.load_model('sentiment_analysis_model.h5')
    return model


model = load_model()

# Load the tokenizer


def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


tokenizer = load_tokenizer()

# General Title
st.title("NLP Sentiment Analysis Project")

# Streamlit UI
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio(
    "Go to", ["Movie Scores", "Test the model!", "Model Training Results"])

if selected_tab == "Movie Scores":
    st.header("Movie Scores")

    # Search box for movie title
    search_term = st.text_input("Search for a movie:")

    # Filter data based on search term
    filtered_data = data[data['Movie'].str.contains(search_term, case=False)]

    # Display filtered data
    if not filtered_data.empty:
        num_movies = len(filtered_data)
        num_columns = 6  # Adjusted to 6 movies per row
        # Ceiling division to calculate number of rows
        num_rows = -(-num_movies // num_columns)

        for i in range(num_rows):
            columns = st.columns(num_columns)
            for j in range(num_columns):
                idx = i * num_columns + j
                if idx < num_movies:
                    movie = filtered_data.iloc[idx]
                    with columns[j]:
                        st.markdown(
                            f"<h3 style='font-size: 14px; line-height: 1.2; max-height: 3.6em; overflow: hidden; text-overflow: ellipsis;'>{movie['Movie']}</h3>", unsafe_allow_html=True)
                        st.image(movie['Poster_URL'], use_column_width=True)
    else:
        st.write("No matching movies found.")

elif selected_tab == "Model Training Results":
    st.write('<iframe title="Sentiment Analysis Test Results" aria-label="Scatter Plot" id="datawrapper-chart-2INNM" src="https://datawrapper.dwcdn.net/2INNM/2/" scrolling="no" frameborder="0" style="width: 80%; min-width: 100% !important; border: none;" height="800" data-external="1"></iframe>', unsafe_allow_html=True)

    st.write('<iframe title="Sentiment Analysis Test Results (Shortened)" aria-label="Scatter Plot" id="datawrapper-chart-P5oxs" src="https://datawrapper.dwcdn.net/P5oxs/2/" scrolling="no" frameborder="0" style="width: 80%; min-width: 100% !important; border: none;" height="800" data-external="1"></iframe>', unsafe_allow_html=True)

else:
    st.subheader("Enter a sentence to analyze its sentiment:")
    user_input = st.text_input("Input Sentence:")

    if user_input:
        # Tokenize and pad the input text
        text_sequence = tokenizer.texts_to_sequences([user_input])
        text_sequence = pad_sequences(text_sequence, maxlen=100)

        # Make a prediction using the trained model
        predicted_rating = model.predict(text_sequence, verbose=None)[0]

        predicted_probabilities = np.array(predicted_rating)
        # print(predicted_rating)

        pos_threshold = 0.9
        neg_threshold = 0.1
        neutral_threshold = 0.99  # Adjust this threshold as needed

        # Calculate the difference between positive and negative probabilities
        diff = abs(predicted_rating[1] - predicted_rating[0])
        # print("Diff: " + str(diff))

        # Check if the difference is below the neutral threshold
        if diff < neutral_threshold:
            predicted_sentiment = 'NEUTRAL'
        # Check if sentiment is positive
        elif predicted_rating[1] > pos_threshold:
            predicted_sentiment = 'POSITIVE'
        # Check if sentiment is negative
        elif predicted_rating[0] > neg_threshold:
            predicted_sentiment = 'NEGATIVE'
        else:
            # Default to neutral if none of the conditions are met
            predicted_sentiment = 'NEUTRAL'

        st.write(
            f"**Predicted Sentiment**: {predicted_sentiment}")
        st.write(
            f"**Negative Sentiment**: {predicted_probabilities[0]:.4f}")
        st.write(
            f"**Positive Sentiment**: {predicted_probabilities[1]:.4f}")

        # Note about sentiment classification
        st.write(
            "\n\n**Note**: Positive/Neutral/Negative classifications can be ambiguous and lose precision in terms of describing the sentiment of text. To account for that, the positive and sentiment scores are provided, which are on a scale from [0, 1] after the final layer of the model performs the softmax function on the sentiment.")
