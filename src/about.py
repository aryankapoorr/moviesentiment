import streamlit as st

def about_page():
    st.markdown("""
    **Note**: all model development can be found in the [project notebook](https://colab.research.google.com/drive/1cl29Xsxy2YjZUaXfmDyi_n9IytefdcLS?usp=sharing)
    and the entire codespace is in the [project repository](https://github.com/aryankapoorr/moviesentiment)
    """)

    st.header("Project Description")
    st.write("""
    As a self-proclaimed cinephile, I am always looking for ways to gather the public's concensus opinion on a movie before watching. 
    All of the current major outlets (Rotten Tomatoes, IMDb, Metacritic, etc.) are decent options, but can be arbitrary with bias and
    varying scoring systems from person to person. Calculating the sentiment of text from a movie review can create a standardized 
    system for understanding the concensus opinion of a movie.
    """)

    st.header("Approach")
    st.write("""
    The general approach of implementing this solution would involve this pipeline:
             
    \tTraining Data Gathering -> Model Training -> Fine Tuning -> Review Scraping -> Movie Scoring -> Database Building
    """)

    st.header("Data Description")
    st.markdown("- Model Training Data: provided courtesy of [Stanford NLP](https://ai.stanford.edu/~amaas/data/sentiment/), containing 50,000 data points with binary outputs (positive/negative). In order to reduce model bias, only 25 reviews per movie were used.")
    st.markdown("- Model Test Results: The outcome of the model test post training is tracked on [this sheet](https://docs.google.com/spreadsheets/d/1OitPcmYJru8GfZj2MEDvHxGfHGd02wo0W03xH_NfWFo/edit?usp=sharing). 10,000 points were used for testing, with 83.75% accuracy.")
    st.markdown("- Movie Score Database: The list of movie scores and poster URLs on the website come from [this sheet](https://docs.google.com/spreadsheets/d/1nEzw584UUzVx7AtWfXxXoRCFXitRhJcou9pDuB-iNco/edit?usp=sharing), which is constantly being updated. Scores are generated off of the top ~250 IMDb reviews")
    st.markdown("- Any intermediary data and the entire process of building the model & gathering data can be found in the [project notebook](https://colab.research.google.com/drive/1cl29Xsxy2YjZUaXfmDyi_n9IytefdcLS?usp=sharing).")

    st.header("Model Intuition")
    st.write("""
    The sentiment analysis model leveras a CNN to gather sentiment of text. After splitting data (80% train 20% test) and 
    initializing the tokenizer, I started building the layers of the neural network.
    - The embedding layer establishes the vocabulary size (with performance being maximized at a size of 5000)
    - A 1D convolutional layer then learns down patterns in the input, breaking down complex relations between the language. This
        step is essential in any natural language model
    - A pooling layer is applied to reduce dimensions and reduce information into a singular vector
    - A fourth dense layer is applied with ReLU as the activation function (sigmoid and tanh were also options but performed worse)
    - A dropout layer then takes place to reduce model overfitting (with the large size of the input, the model was very susceptible to
             overfitting, making this layer crucial)
    - The final layer performs softmax to normalize results in the range [0, 1]
    """)
    st.latex(r'''\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}''')
    st.write("""
             - Adam was the best performing optimizer. Tests were conducted with AdaGrad, but the model was taking too long to converge
             - After testing serveral epoch/batch size combinations, model fitting performance was maximized with a batch size of 530 and 13 epochs
             """)

    st.header("Scoring Intuition")
    st.write("""
    After looking at the distribution of data points from the model training results, data was not clustering in a way that would allow 
    me to easily distinguish between positive and negative reviews. Therefore, in order to not lose the precision of the exact sentiment scores,
    I took the weighted average of the sentiment scores, instead of converting them into a binary outcome (positive/negative) first.
    
    To address the clustering of data at the extreme ends, I implemented a weighting system that would value scores on the more
    extreme ends of the spectrum. In order to do this, I used the gaussian function to distribute data points accordingly
    """)
    st.latex(r'''f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}''')

    st.write("""
    However, in this case we would want the inverse effect of a normal distribution, to add more weight to extreme values, which would be the probit function
    """)
    st.latex(r'''\Phi^{-1}(x) =\text{inverse\_gaussian\_weight}(x, \mu=0.5, \sigma=0.1) = 1 - e^{-0.5 \left(\frac{{x - \mu}}{{\sigma}}\right)^2}''')
    st.write("After normalizing the weights and taking the weighted average, I was then able to calculate the overall score for a movie's sentiment")

    st.header("About the Author")
    image_path = "./data/authPhoto.jpeg"
    col1, col2 = st.columns(2)  # Adjust the width ratio as needed
    with col1:
        # Display the image with a smaller width
        st.image(image_path)

    with col2:
        # Display text to the right of the image
        st.write("Hello! I'm Aryan, graduating from UT Dallas in 2025 with a Masters in Sciences in Computer Science, concentrating on Intelligent Systems and Data Science.")
    
        github_url = "https://github.com/aryankapoorr"
        linkedin_url = "https://www.linkedin.com/in/aryan-kapoor/"

        # Display links with icons to GitHub and LinkedIn
        st.markdown(f"<a href='{github_url}' target='_blank'><img src='https://img.icons8.com/fluent/48/000000/github.png' width='25' style='vertical-align: bottom'></a> [GitHub]({github_url})", unsafe_allow_html=True)
        st.markdown(f"<a href='{linkedin_url}' target='_blank'><img src='https://img.icons8.com/fluent/48/000000/linkedin.png' width='25' style='vertical-align: bottom'></a> [LinkedIn]({linkedin_url})", unsafe_allow_html=True)
