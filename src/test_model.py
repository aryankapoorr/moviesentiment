import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the Keras model and tokenizer
# Define load_model() and load_tokenizer() functions here

def test_model_page():
    def load_model():
        # Assuming the model file is 'sentiment_analysis_model.h5'
        model = tf.keras.models.load_model('./src/sentiment_analysis_model.h5')
        return model


    model = load_model()

    # Load the tokenizer
    def load_tokenizer():
        with open('./src/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer


    tokenizer = load_tokenizer()

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
        # diff = abs(predicted_rating[1] - predicted_rating[0])
        # print("Diff: " + str(diff))
        s = np.argmax(predicted_probabilities)

        # Check if the difference is below the neutral threshold
        if s == 1:
            predicted_sentiment = 'POSITIVE'
        else:
            predicted_sentiment = 'NEGATIVE'

        st.write(
            f"**Predicted Sentiment**: {predicted_sentiment}")
        st.write(
            f"**Negative Sentiment**: {predicted_probabilities[0]:.4f}")
        st.write(
            f"**Positive Sentiment**: {predicted_probabilities[1]:.4f}")

        # Note about sentiment classification
        st.write(
            "\n\n**Note**: Positive/Negative classifications can be ambiguous and lose precision in terms of describing the sentiment of text. To account for that, the positive and negative sentiment scores are provided, which are on a scale from [0, 1] after the final layer of the model performs the softmax function on the sentiment.")
        st.write("\n\n**Additionally**, the model is tuned specifically to movie reviews. Content put into the model tester that does not meet the appropriate length (250 characters) or subject matter can lead to inaccurate results.")
