import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ------------------ SPOTIFY API ------------------
CLIENT_ID = os.getenv("CLIENT_ID") or "95bfc5512755496c8cc59b0ada0c440e"
CLIENT_SECRET = os.getenv("CLIENT_SECRET") or "67df5477e9634201bfc81f5e3fb17e03"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
))

# ------------------ FETCH POSTER ------------------
def get_song_album_cover_url(song_name, artist_name):
    try:
        query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=query, type="track")

        if results and results["tracks"]["items"]:
            return results["tracks"]["items"][0]["album"]["images"][0]["url"]
        else:
            return "https://i.postimg.cc/0QNxYz4V/social.png"
    except:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

# ------------------ LOAD DATA ------------------
music = pickle.load(open('df.pkl', 'rb'))

# Clean song names
music['song'] = music['song'].str.lower().str.strip()

# ------------------ COMPUTE SIMILARITY ------------------
@st.cache_data
def compute_similarity(data):
    tfidf = TfidfVectorizer(max_features=2000)
    vectors = tfidf.fit_transform(data['text']).toarray()
    return cosine_similarity(vectors)

similarity = compute_similarity(music)

# ------------------ RECOMMEND FUNCTION ------------------
def recommend(song):
    song = song.lower().strip()
    matches = music[music['song'] == song]

    if matches.empty:
        return [], []

    index = matches.index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    names = []
    posters = []

    for i in distances[1:6]:
        song_name = music.iloc[i[0]].song
        artist = music.iloc[i[0]].artist

        names.append(song_name)
        posters.append(get_song_album_cover_url(song_name, artist))

    return names, posters

# ------------------ STREAMLIT UI ------------------
st.title("🎵 Music Recommender System")

music_list = music['song'].values

selected_song = st.selectbox(
    "Select a song",
    music_list
)

if st.button("Recommend"):
    names, posters = recommend(selected_song)

    if not names:
        st.error("Song not found")
    else:
        cols = st.columns(5)

        for i in range(5):
            with cols[i]:
                st.text(names[i])
                st.image(posters[i])