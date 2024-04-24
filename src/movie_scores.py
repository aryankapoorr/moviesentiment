import streamlit as st
import pandas as pd

@st.cache_data
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

def movie_scores_page():
    st.header("Movie Scores")

    # Load the CSV file
    data = load_data('./data/Sentiment Analysis Data - Sheet1.csv')

    # Search box for movie title
    search_term = st.text_input("Search for a movie:")

    # Sorting options
    sort_options = {
        "Highest Score": "Score",
        "Lowest Score": "Score",
        "Alphabetically (A-Z)": "Movie",
        "Alphabetically (Z-A)": "Movie [reverse]"
    }
    selected_sort_option = st.selectbox("Sort by", list(sort_options.keys()))

    # Sorting logic
    if selected_sort_option == "Highest Score":
        ascending = False
    elif selected_sort_option == "Lowest Score":
        ascending = True
    elif selected_sort_option == "Alphabetically (A-Z)":
        ascending = True
    elif selected_sort_option == "Alphabetically (Z-A)":
        ascending = False

    if "[reverse]" in selected_sort_option:
        selected_sort_option = selected_sort_option.replace(
            "[reverse]", "").strip()
        sorted_data = data.sort_values(
            by=sort_options[selected_sort_option], ascending=ascending).iloc[::-1]
    else:
        sorted_data = data.sort_values(
            by=sort_options[selected_sort_option], ascending=ascending)

    # Filter data based on search term
    filtered_data = sorted_data[sorted_data['Movie'].str.contains(
        search_term, case=False)]

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
                            f"<h3 style='font-size: 14px; line-height: 1.2; max-height: 3.6em; overflow: hidden; text-overflow: ellipsis; text-align: center;'>{movie['Movie']}</h3>", unsafe_allow_html=True)
                        st.image(
                            movie['Poster_URL'], caption=f"Score: {movie['Score']}", use_column_width=True)
    else:
        st.write("No matching movies found.")
