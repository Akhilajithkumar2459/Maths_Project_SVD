import streamlit as st
from surprise import SVD
import pandas as pd
import pickle

# --- Load Movie Mapping Dataset ---
@st.cache_data
def load_movie_data():
    return pd.read_csv('movie.csv')

@st.cache_data
def load_ratings_data():
    return pd.read_csv('rating.csv')  # Assuming you have a 'ratings.csv' with user ratings data
    

movies_df = load_movie_data()
ratings_df = load_ratings_data()

# --- Streamlit App ---
st.title("🎬 Movie Recommendation System")
st.markdown("A movie recommendation app using a **pre-trained SVD model**.")

# Add background image using HTML and CSS
st.markdown("""
    <style>
        .stApp {
            background-image: url("p4.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            min-height: 100vh;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.7);
            padding: 20px;
            border-radius: 10px;
        }
        .recommend-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .recommend-button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Step 1: Load Pre-trained Model
with st.spinner("Loading pre-trained model..."):
    try:
        with open("svd_model.pkl", "rb") as f:
            model = pickle.load(f)
        st.success("Pre-trained model loaded successfully!")
        model_loaded = True
    except FileNotFoundError:
        st.error("Pre-trained model file not found. Make sure `svd_model.pkl` is in the same directory.")
        model_loaded = False

if model_loaded:
    # Organizing UI using tabs
    tab1, tab2 = st.tabs(["🔍 Input", "🎥 Recommendations"])

    # Step 2: User Input Section
    with tab1:
        st.header("Select a Movie and Enter Your User ID")

        # Input field for User ID with a placeholder
        user_id = st.number_input("Enter Your User ID", min_value=1, step=1, help="E.g., 1, 2, 3...")

        # Validate User ID existence
        valid_user_ids = ratings_df['userId'].unique()
        if user_id not in valid_user_ids:
            st.warning("User ID not found in the database. Please try a different User ID.")

        # Movie selection dropdown with search functionality
        selected_movie = st.selectbox("Select a Movie", movies_df['title'].values, help="Select a movie from the list")

    # Step 3: Recommendations Section
    with tab2:
        if selected_movie and st.button("Recommend Movies", key="recommend_button"):
            with st.spinner("Generating recommendations..."):
                try:
                    # Find the movie ID of the selected movie
                    movie_id = movies_df[movies_df['title'] == selected_movie]['movieId'].values[0]

                    # Get all movies the user has already rated
                    rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].values

                    # Predict ratings for all movies that the user has not rated yet
                    all_movie_ids = movies_df['movieId'].values
                    unrated_movie_ids = [movie for movie in all_movie_ids if movie not in rated_movies]

                    predictions = [
                        (movie, model.predict(user_id, movie).est)
                        for movie in unrated_movie_ids
                    ]

                    # Convert predictions to a DataFrame
                    predictions_df = pd.DataFrame(predictions, columns=["movieId", "predicted_rating"])

                    # Merge with movie titles and genres
                    predictions_df = predictions_df.merge(movies_df, on="movieId")
                    predictions_df = predictions_df.sort_values(by="predicted_rating", ascending=False)

                    # Filter top 5 movies with higher ratings
                    top_movies = predictions_df.head(5)

                    # Display recommended movies
                    if not top_movies.empty:
                        st.success("Top 5 Recommended Movies for You:")
                        for _, row in top_movies.iterrows():
                            st.write(f"🎥 **{row['title']}** (Rating: {row['predicted_rating']:.2f})")
                            st.text(f"Genre: {row.get('genres', 'N/A')}")  # Display genre if available
                    else:
                        st.warning("No related movies found.")
                except Exception as e:
                    st.error(f"Error in generating recommendations: {str(e)}")
