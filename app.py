import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

CLIENT_ID = "672e6bcfc573440c8621425328e2b404"
CLIENT_SECRET = "914926df1bd94ce4a1f2f2aa12615e85"

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# Function to get album cover URL
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"


# Load your dataset
df = pd.read_csv("cleaned_dataset.csv")


# Data Preprocessing function
def preprocess_data(df):
    df.dropna(inplace=True)
    columns_to_drop = ['Artist', 'Album', 'Album_type', 'Title', 'Channel', 'Licensed', 'official_video', 'most_playedon']
    dropped_df = df.drop(columns=columns_to_drop)
    numerical_columns = ['Loudness', 'Tempo', 'Duration_min', 'Views', 'Likes', 'Comments', 'Energy', 'Liveness', 'Stream', 'EnergyLiveness']
    dropped_df[numerical_columns] = (dropped_df[numerical_columns] - dropped_df[numerical_columns].min()) / (dropped_df[numerical_columns].max() - dropped_df[numerical_columns].min())
    return dropped_df


# Function to find track index
def find_track_index(track_name, df):
    if 'Track' not in df.columns:
        raise ValueError("DataFrame does not have a 'Track' column")
    try:
        track_index = df.loc[df['Track'] == track_name].index[0]
        return track_index
    except IndexError:
        return None


# Function to find song recommendations
def find_song_recommendation(track_name, df, num_recommendations=5):
    track_index = find_track_index(track_name, df)
    if track_index is None:
        st.write(f"Track '{track_name}' not found in the DataFrame")
        return None
    cluster = df.loc[track_index, 'Cluster']
    filtered_df = df[df['Cluster'] == cluster]
    if filtered_df.empty:
        st.write("No similar tracks found in the same cluster")
        return None
    recommendations = filtered_df.sample(num_recommendations)

    # Fetch album cover URLs
    album_covers = []
    for index, row in recommendations.iterrows():
        cover_url = get_song_album_cover_url(row['Track'], row['Artist'])
        album_covers.append(cover_url)

    recommendations['Album Cover URL'] = album_covers
    return recommendations


# Main function to run the app
def main():
    st.title("Song Recommendation System")
    st.write("This app recommends songs based on your favorite track.")

    # Preprocess data
    processed_df = preprocess_data(df)
    numeric_df = processed_df.select_dtypes(include=[np.number])

    # Clustering
    kmeans = KMeans(n_clusters=3, n_init='auto')
    kmeans.fit(numeric_df)
    df['Cluster'] = kmeans.labels_
    numeric_df['Cluster'] = kmeans.labels_

    # User input
    track_name = st.text_input("Enter a track name to get recommendations:")
    num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)

    if st.button("Get Recommendations"):
        recommendations = find_song_recommendation(track_name, df, num_recommendations)
        if recommendations is not None:
            st.write("Recommended Tracks:")

            # Display recommendations in rows with at most 4 columns
            num_tracks = len(recommendations)
            num_cols = 4

            for i in range(0, num_tracks, num_cols):
                row = recommendations.iloc[i:i + num_cols]

                # Create columns for each track (maximum of 4 per row)
                cols = st.columns(num_cols)

                for idx, (_, row_data) in enumerate(row.iterrows()):
                    track = row_data['Track']
                    artist = row_data['Artist']
                    cover_url = row_data['Album Cover URL']

                    with cols[idx]:
                        st.image(cover_url, caption=f"{track} by {artist}", width=150)

                    if idx == 0:
                        cols[idx].markdown('---')  # Add a separator between rows

if __name__ == "__main__":
    main()
