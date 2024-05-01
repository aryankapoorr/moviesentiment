# Movie Sentiment Analysis Project
<img src="https://github.com/aryankapoorr/moviesentiment/blob/main/data/deniro.webp" width=300>

For a live demonstration of the sentiment analysis tool, visit the [Movie Sentiment Analyzer](https://moviesentiment.streamlit.app/)

Explore the [Project Notebook](https://colab.research.google.com/drive/1cl29Xsxy2YjZUaXfmDyi_n9IytefdcLS?usp=sharing) for an in-depth look at implementation

## Project Description
As a self-proclaimed cinephile, I am always looking for ways to gather the public's concensus opinion on a movie before watching. All of the current major outlets (Rotten Tomatoes, IMDb, Metacritic, etc.) are decent options, but can be arbitrary with bias and varying scoring systems from person to person. Calculating the sentiment of text from a movie review can create a standardized system for understanding the concensus opinion of a movie.

After finding an already-cleaned database of movie reviews, I developed a Sentiment Analysis Model using a Convolutional Neural Network. After testing various paramaters and epochs of model training, I fit the model and created a scoring system, weighting the sentiments based on the inverse of their gaussian distribution. Then, ~250 reviews were collected per movie and passed through the model, with each score being uploaded to the website.

## Demo
[Demo](#) - Placeholder for the demo link. To be updated.

## Usage
To use the sentiment analysis tool locally, follow these steps:
1. Clone the project notebook
2. Go through all of the cells in the **_model building_** section, making sure there are no errors
3. Go through all of the cells in the **_model testing and scoring_** section, making sure there are no errors
4. Connect your google drive in the appropriate cells in the **_database building_** section
5. Edit the **_names_** variable to the movie names of your choice
6. Run the rest of the cells in the **_database building_** section.

## Data Description

- **Model Training Data:** Provided courtesy of [Stanford NLP](https://ai.stanford.edu/~amaas/data/sentiment/), containing 50,000 data points with binary outputs (positive/negative). In order to reduce model bias, only 25 reviews per movie were used.
- **Model Test Results:** The outcome of the model test post training is tracked on [this sheet](https://docs.google.com/spreadsheets/d/1OitPcmYJru8GfZj2MEDvHxGfHGd02wo0W03xH_NfWFo/edit?usp=sharing). 10,000 points were used for testing, with 83.75% accuracy.
- **Movie Score Database:** The list of movie scores and poster URLs on the website come from [this sheet](https://docs.google.com/spreadsheets/d/1nEzw584UUzVx7AtWfXxXoRCFXitRhJcou9pDuB-iNco/edit?usp=sharing), which is constantly being updated. Scores are generated off of the top ~250 IMDb reviews.
- **Project Notebook:** Any intermediary data and the entire process of building the model & gathering data can be found in the [project notebook](https://colab.research.google.com/drive/1cl29Xsxy2YjZUaXfmDyi_n9IytefdcLS?usp=sharing).


## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- Special thanks to [Stanford NLP](https://ai.stanford.edu/~amaas/data/sentiment/) for allowing the use of their dataset for model training.
- Thanks to [Streamlit](https://streamlit.io/) for providing a fantastic platform for building interactive web apps.
- Thanks to [Google Colab](https://colab.research.google.com/) for providing a free and powerful environment for running Jupyter notebooks.

