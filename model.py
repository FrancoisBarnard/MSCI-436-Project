import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

"""
NOTE:
The algorithm and a large part of the model implementation is based off of
Kaggle user "Niharika Pandit"'s notebook at
https://www.kaggle.com/code/niharika41298/netflix-visualizations-recommendation-eda
"""


def main():
    data, untouched_data = process_data()

    # Identifying features for the model
    features = ["title", "director", "cast", "description"]
    data = data[features]

    # Apply cleaning function
    for feature in features:
        data[feature] = data[feature].apply(clean_data)

    # Create soup of the data
    data["soup"] = data.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words="english")

    count_matrix = count.fit_transform(data["soup"])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    data = data.reset_index()
    indices = pd.Series(data.index, index=data["title"])

    # Streamlit configuring
    st.title("Streaming Platform Recomendation System")

    # Create a list of titles for the user to search
    title_list = untouched_data["title"].values

    # Selectable dropdown for th user
    with st.spinner("Processing..."):
        selected_title = st.selectbox(
            "Type or manually select the movie or TV show from the dropdown", title_list)

    # Create a button for the user to get the recommendation
    if st.button("Show Recommendation"):
        with st.spinner("Processing..."):
            recommendations = get_recommendations(
                selected_title, cosine_sim, indices, untouched_data)
            final = get_best_service(recommendations, untouched_data)

            # Get sorted list of results
            result = sorted(final, key=final.get, reverse=True)

            # display results
            st.header("Based on your input you should get...")
            for idx, service in enumerate(result):
                st.subheader(f"{idx + 1}: {service.capitalize()}")

            with st.expander("Advanced details"):
                final


def process_data():
    """
    Handle the initial data processing for the model
    """
    # Import data
    netflix = pd.read_csv("data/netflix_titles.csv")
    amazon = pd.read_csv("data/amazon_prime_titles.csv")
    hulu = pd.read_csv("data/hulu_titles.csv")
    disney = pd.read_csv("data/disney_plus_titles.csv")

    # Combining the initial data without any labels for origin
    data = pd.concat([netflix, amazon, hulu, disney], ignore_index=True)

    # Removing "na" fields
    data = data.fillna("")

    # Removing any duplicate movies
    data = data.drop_duplicates(subset=["title", "type"], keep="last")

    # Adding a label to all the data with their source
    netflix = netflix.assign(netflix=1, amazon=0, hulu=0, disney=0)
    amazon = amazon.assign(netflix=0, amazon=1, hulu=0, disney=0)
    hulu = hulu.assign(netflix=0, amazon=0, hulu=1, disney=0)
    disney = disney.assign(netflix=0, amazon=0, hulu=0, disney=1)

    # Defining aggregate functions
    g = {"netflix": "sum", "amazon": "sum", "hulu": "sum", "disney": "sum"}

    # Creating a temporary dataframe with only titles, type, & origins
    data_temp = pd.concat([netflix, amazon, hulu, disney], ignore_index=True).groupby(
        ["title", "type"], as_index=False).agg(g).reset_index()
    data_temp = data_temp.fillna("")

    # Merging the two dataframes together to create a complete system
    # Inner merge information found via TutorialsPoint tutorial
    # Source: https://www.tutorialspoint.com/python_pandas/python_pandas_merging_joining.htm
    data = pd.merge(data, data_temp, on=["title", "type"], how="inner")

    untouched_data = data

    return data, untouched_data


def clean_data(data):
    """
    Define a function to clean the data by forcing all needed groups lowercase
    """
    return str.lower(data.replace(" ", ""))


def create_soup(data):
    """
    Create a "soup" or "bag of words" for all rows
    """
    output = data["title"] + " " + data["director"] + " " + \
        data["cast"] + " " + " " + data["description"]

    return output


def get_recommendations(title, cosine_sim, indices, output_df):
    """
    Get recommendations based off of input movie title
    """
    title = title.replace(' ', '').lower()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return output_df['title'].iloc[movie_indices]


def get_best_service(recommendation, output_df):
    """
    Returns the count of where the recommendations can be found
    """
    output = {
        "Netflix": 0,
        "Amazon Prime Video": 0,
        "Hulu": 0,
        "Disney+": 0
    }
    print(recommendation)
    for title in recommendation:
        output["Netflix"] += int(output_df.loc[output_df["title"]
                                 == title, "netflix"].values[0])
        output["Amazon Prime Video"] += int(output_df.loc[output_df["title"]
                                                          == title, "amazon"].values[0])
        output["Hulu"] += int(output_df.loc[output_df["title"]
                              == title, "hulu"].values[0])
        output["Disney+"] += int(output_df.loc[output_df["title"]
                                               == title, "disney"].values[0])

    return output


if __name__ == "__main__":
    main()
