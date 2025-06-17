import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import io

# Set Streamlit page configuration for a wider layout and a pleasant theme
st.set_page_config(layout="wide", page_title="🎶 Song Recommender", page_icon="🎵")

# Custom CSS for a more interesting and user-friendly look
st.markdown("""
    <style>
    .main {
        background-color: #1a1a2e; /* Dark blue background */
        color: #e0e0e0; /* Light grey text */
    }
    .stSelectbox > div > div {
        background-color: #33334d; /* Darker blue for select box */
        color: #e0e0e0;
        border-radius: 10px;
        border: 1px solid #4a4a6e;
    }
    .stSlider > div > div > div {
        background-color: #4a4a6e; /* Slider track color */
        border-radius: 5px;
    }
    .stSlider .st-bd {
        color: #e0e0e0; /* Slider text color */
    }
    .stButton > button {
        background-color: #e94560; /* Pinkish-red for button */
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 1.1em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .stButton > button:hover {
        background-color: #b82b43; /* Darker pink on hover */
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #4a4a6e;
    }
    .css-1d391kg { /* Target for header color if needed */
        color: #e94560;
    }
    h1 {
        color: #e94560; /* Main title color */
        text-align: center;
        font-size: 3.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    h2 {
        color: #e0e0e0; /* Section titles */
        font-size: 2em;
        margin-top: 30px;
        border-bottom: 2px solid #4a4a6e;
        padding-bottom: 10px;
    }
    .css-fg4lbf p { /* Paragraph text color */
        color: #e0e0e0;
    }
    .recommendation-box {
        background-color: #33334d; /* Darker blue for recommendation boxes */
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #4a4a6e;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .recommendation-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #e94560; /* Pinkish-red for song title */
    }
    .recommendation-artist {
        font-size: 1em;
        color: #e0e0e0;
    }
    .recommendation-similarity {
        font-size: 0.9em;
        color: #a0a0a0; /* Lighter grey for similarity */
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎶 Personalized Song Recommender")
st.markdown("### Discover your next favorite track!")

@st.cache_data # Cache data loading and preprocessing to avoid re-running on every interaction
def load_and_preprocess_data():
    """
    Loads the Spotify dataset, preprocesses it, trains the autoencoder,
    and calculates the similarity matrix.
    """
    try:
        # Assuming 'Spotify-2000.csv' is in the same directory as the app.py
        file_path = 'Spotify-2000.csv'
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. Please ensure the CSV file is in the same directory.")
        st.stop() # Stop execution if file is not found

    # Drop the 'Index' column if it exists, as it's not a feature
    if 'Index' in df.columns:
        df = df.drop('Index', axis=1)

    # Feature columns for the autoencoder
    feature_columns = [
        'Year', 'Beats Per Minute (BPM)', 'Energy', 'Danceability', 'Loudness (dB)',
        'Liveness', 'Valence', 'Acousticness', 'Speechiness', 'Popularity'
    ]

    # Handle missing values by filling with the mean (or other suitable strategy)
    for col in feature_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    X = df[feature_columns].values

    # Check for NaNs/Infs and handle them before scaling
    X = np.nan_to_num(X)

    # Normalize data between 0 and 1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Autoencoder Model Definition and Training
    input_dim = X_scaled.shape[1]
    encoding_dim = 64 # Reduced dimension for embeddings

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded) # sigmoid for scaled data

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    # Train the autoencoder (using all data for embeddings for simplicity in app)
    # In a real scenario, you'd train on X_train and then get embeddings for all X
    with st.spinner("Training autoencoder and generating embeddings... This might take a moment."):
        autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)
    st.success("Autoencoder training complete!")

    # Define the encoder model to extract embeddings
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Get the song embeddings from the trained encoder
    song_embeddings = encoder.predict(X_scaled, verbose=0)

    # Calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(song_embeddings)

    return df, scaler, encoder, similarity_matrix, feature_columns

# Load data and models
df, scaler, encoder, similarity_matrix, feature_columns = load_and_preprocess_data()

# Recommendation Logic
def get_recommendations(song_title, df, similarity_matrix, num_recommendations=5):
    """
    Finds songs similar to the given song title and returns them with similarity scores.
    """
    if song_title not in df['Title'].values:
        return [] # Return an empty list to indicate not found

    # Get the index of the song that matches the title
    idx = df[df['Title'] == song_title].index[0]

    # Get the similarity scores for this song with all other songs
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort the songs based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the `num_recommendations` most similar songs (excluding the song itself)
    sim_scores = sim_scores[1:num_recommendations+1]

    recommended_songs_data = []
    for i, score in sim_scores:
        song_info = df.iloc[i]
        recommended_songs_data.append({
            'Title': song_info['Title'],
            'Artist': song_info['Artist'],
            'Similarity': score
        })
    return recommended_songs_data

# Streamlit App Layout
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Select a Song")
    selected_song = st.selectbox(
        "Choose a song from the list to get recommendations:",
        df['Title'].sort_values().unique(),
        key="song_select"
    )

with col2:
    st.subheader("Recommendation Preferences")
    num_recommendations = st.slider(
        "How many recommendations would you like?",
        min_value=1,
        max_value=20,
        value=5,
        key="num_recommendations"
    )

st.markdown("---")

if st.button("Get Recommendations"):
    if selected_song:
        st.markdown(f"<h2>Your Top Recommendations for: <span style='color:#e94560;'>{selected_song}</span></h2>", unsafe_allow_html=True)
        with st.spinner("Fetching recommendations..."):
            recommendations = get_recommendations(selected_song, df, similarity_matrix, num_recommendations)

            if recommendations:
                for i, rec_song in enumerate(recommendations):
                    st.markdown(f"""
                        <div class="recommendation-box">
                            <div class="recommendation-title">{i+1}. {rec_song['Title']} by {rec_song['Artist']}</div>
                            <div class="recommendation-similarity">Similarity: {rec_song['Similarity']:.4f}</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Song not found in the dataset. Please select from the list.")
    else:
        st.warning("Please select a song to get recommendations.")

st.markdown("---")
st.markdown("#### How it works:")
st.markdown("""
This app uses a **deep learning autoencoder** to learn meaningful numerical representations (embeddings) of songs based on their audio features (BPM, Energy, Danceability, etc.).
Once the embeddings are generated, we use **cosine similarity** to find songs that have similar embeddings, implying they are musically similar.
""")
