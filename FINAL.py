# MoodBot: Cinema-Themed Streamlit App

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


# Load data
df_train = pd.read_csv("training.csv")
df_movies = pd.read_csv("all_movies_df.csv")
df_food = pd.read_csv("Food_final.csv")

# Emotion label mapping
label_to_emotion = {0: "sadness", 1: "joyful", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
df_train["emotion"] = df_train["label"].map(label_to_emotion)

# Train emotion detection model
X_train = df_train["text"]
y_train = df_train["emotion"]
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ("clf", LogisticRegression(max_iter=300))
])
model.fit(X_train, y_train)

# Genre to mood mapping
genre_to_mood = {
    "Romance": "love", "Comedy": "joyful", "Action": "anger",
    "Drama": "sadness", "Horror": "fear", "Fantasy": "surprise",
    "Musical": "joyful", "Adventure": "surprise"
}

def map_genres_to_moods(genre):
    if pd.isna(genre): return []
    genres = re.split(r'[|,]', genre)
    return [genre_to_mood[g.strip()] for g in genres if g.strip() in genre_to_mood]

df_movies["mood_list"] = df_movies["genre"].apply(map_genres_to_moods)

# Preprocess food dataset
df_food['TotalMinutes'] = pd.to_numeric(df_food['TotalTime'], errors='coerce')
df_food['text'] = df_food[['Name', 'Ingredients', 'Keywords', 'RecipeCategory']].fillna('').agg(' '.join, axis=1)
df_food['Images'] = df_food['Images'].apply(lambda x: x if isinstance(x, str) and x.startswith('http') else None)

# Fit vectorizer
food_vectorizer = TfidfVectorizer(stop_words='english')
food_vectorizer.fit(df_food['text'])

# Cinema theme UI
st.set_page_config(page_title="MoodBot", page_icon="🎬", layout="centered")
st.markdown("""
    <style>
        body, .main {
            background: linear-gradient(to bottom, #000000, #1c1c1c);
            color: #f8f8f2;
        }
        .stTextInput > div > div > input,
        .stTextArea > div > textarea {
            background-color: #333;
            color: white;
        }
        .stButton > button {
            background-color: #e50914;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .movie-card {
            background-color: #121212;
            border-left: 5px solid #e50914;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .recipe-card {
            background-color: #262626;
            border-left: 5px solid #f5c518;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: #f5c518;'>
    <span class='popcorn'>🍿</span> MoodBot Cinema Edition
    </h1>
    <style>
    .popcorn {
        display: inline-block;
        animation: bounce 0.8s infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    </style>

    <p style='text-align: center; color: #ccc;'>Your personal mood-based movie & snack recommender!</p>
    <hr style='border-top: 2px dashed #e50914;'>
""", unsafe_allow_html=True)

user_input = st.text_input("💬 What’s your mood today?")

if user_input:
    emotion = model.predict([user_input])[0]
    emoji_map = {
        "joyful": "🎉", "sadness": "😭", "love": "💖",
        "anger": "🔥", "fear": "😱", "surprise": "🤯"
    }
    st.markdown(f"<h3 style='color:#f5c518;'>🎭 Mood Detected: {emoji_map[emotion]} {emotion.capitalize()}</h3>", unsafe_allow_html=True)

    # 🎵 Music Recommendations Section
    st.subheader("🎶 Music Recommendations")

    music_recs = {
        "joyful": [
            ("Happy – Pharrell Williams", "https://www.youtube.com/watch?v=ZbZSe6N_BXs"),
            ("Can't Stop the Feeling – Justin Timberlake", "https://www.youtube.com/watch?v=ru0K8uYEZWw")
        ],
        "sadness": [
            ("Let Her Go – Passenger", "https://www.youtube.com/watch?v=RBumgq5yVrA"),
            ("Someone Like You – Adele", "https://www.youtube.com/watch?v=hLQl3WQQoQ0")
        ],
        "love": [
            ("Perfect – Ed Sheeran", "https://www.youtube.com/watch?v=2Vv-BfVoq4g"),
            ("All of Me – John Legend", "https://www.youtube.com/watch?v=450p7goxZqg")
        ],
        "anger": [
            ("In The End – Linkin Park", "https://www.youtube.com/watch?v=eVTXPUF4Oz4"),
            ("Stronger – Kanye West", "https://www.youtube.com/watch?v=PsO6ZnUZI0g")
        ],
        "fear": [
            ("Creep – Radiohead", "https://www.youtube.com/watch?v=XFkzRNyygfk"),
            ("Everybody Hurts – R.E.M.", "https://www.youtube.com/watch?v=ijZRCIrTgQc")
        ],
        "surprise": [
            ("On Top of the World – Imagine Dragons", "https://www.youtube.com/watch?v=w5tWYmIOWGk"),
            ("Best Day of My Life – American Authors", "https://www.youtube.com/watch?v=Y66j_BUCBMY")
        ]
    }

    for title, link in music_recs.get(emotion, []):
        st.markdown(f"""
            <div style='background-color:#121212; border-left: 5px solid #e50914;
                        padding: 12px 20px; border-radius: 10px; margin-bottom: 15px;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.3);'>
                <h4 style='color:#f5c518; margin-bottom: 6px;'>{title}</h4>
                <a href='{link}' target='_blank' style='color:#e50914;'>▶️ Listen Now</a>
            </div>
        """, unsafe_allow_html=True)

    # (Movie + food recommendation code remains here inside the block)

    matched_movies = df_movies[df_movies["mood_list"].apply(lambda moods: emotion in moods)]
    if not matched_movies.empty:
        st.subheader("🎬 Movie Picks Just for You:")
        for _, row in matched_movies.sample(n=min(3, len(matched_movies))).iterrows():
            st.markdown(f"""
                <div class='movie-card'>
                    <div style='display: flex; align-items: center;'>
                        <img src='{row['poster_link']}' alt='poster' style='width:100px; margin-right:15px; border-radius:10px;' />
                        <div>
                            <h4 style='color:#f5c518; margin-bottom: 5px;'>{row['title']}</h4>
                            <p style='color:#ccc; margin: 0;'>🎞️ Genre: {row['genre']}</p>
                            <a href='{row['movie_link']}' target='_blank' style='color:#e50914;'>▶️ Watch Trailer</a>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    food_input = st.text_input("🍔 Any food cravings? Describe what you want or say 'no':")
    if food_input:
        if food_input.strip().lower() == 'no':
            st.success("🎬 Grab your popcorn and enjoy the movie!")
        else:
            input_vec = food_vectorizer.transform([food_input])
            food_tfidf = food_vectorizer.transform(df_food['text'])
            sim_scores = cosine_similarity(input_vec, food_tfidf).flatten()
            top_indices = sim_scores.argsort()[-5:][::-1]
            top_foods = df_food.iloc[top_indices]

            if top_foods.empty or max(sim_scores) < 0.1:
                st.warning("🍽️ Hmm, couldn't find a match. Try describing it differently?")
            else:
                st.subheader("🍕 Movie Night Snacks:")
                for _, row in top_foods.iterrows():
                    if row['Images']:
                        st.image(row['Images'], width=150)
                    st.markdown(f"""
                        <div class='recipe-card'>
                            <h4 style='color:#f5c518;'>{row['Name']}</h4>
                            <p style='color:#bbb;'>⏱️ {row['TotalTime']} mins<br>📝 {row['Ingredients']}</p>
                            <a href='{row['RecipeUrl']}' target='_blank' style='color:#f5c518;'>📖 View Recipe</a>
                        </div>
                    """, unsafe_allow_html=True)
