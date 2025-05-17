
# MoodBot 🎬🍿 – Emotion-Based Movie, Music & Food Recommender

This is a cinema-themed AI project built using Streamlit. It detects the user's emotion based on natural language input and recommends:
- 🎥 Movies that match the detected emotion
- 🎵 Music tracks to vibe with your mood
- 🍕 Recipes or snacks based on optional cravings

---

## 👩🏻‍💻 Author
**Shreya Gupta**  
**SBU ID:** 116742821

---

## 📁 Project Structure

```
.
├── FINAL.py                         # Main Streamlit app
├── training.csv                  # Emotion-labeled training dataset
├── validation.csv                # Optional validation data
├── test.csv                      # Test samples
├── all_movies_df.csv            # Movie metadata (titles, genres, links, posters)
├── Food_final.csv               # Food recipes and metadata
```

---

## ⚙️ Setup Instructions

### 1. Clone or download the repository
```bash
git clone https://github.com/Shreyagupta-sg/MoodBot
cd MoodBot
```

### 2. Install the required dependencies
It is recommended to use a virtual environment (optional):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt, manually install:

```bash
pip install streamlit pandas scikit-learn matplotlib
```

---

## 🚀 How to Run the App

Make sure all your data files (CSV) are in the same directory as ani.py.

Run the app using Streamlit:

```bash
streamlit run FINAL.py
```

This will open the app in your default browser (usually at http://localhost:8501).

---

## 💡 Features

- Detects emotional intent from text using a trained Logistic Regression model
- Maps genres to moods for movie recommendations
- Matches food craving text with recipe data using cosine similarity
- Provides fallback images if posters or food images are missing
- Cinema-themed UI with animated popcorn and styled input fields

---

## 📬 Contact

For any questions, reach out to:  
📧 shreya.gupta@stonybrook.edu 
