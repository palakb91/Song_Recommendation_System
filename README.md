# 🎶 Personalized Song Recommender

A visually engaging and intelligent **Streamlit web app** that recommends songs based on user selection using **autoencoder-based deep learning** and **cosine similarity**. Discover musically similar tracks using audio features like BPM, Danceability, Energy, and more!

---

## 🚀 Live Demo

👉 [Try it Now](https://songrecommendationsystem-5ag7prcu7jvh95gimmmf88.streamlit.app/)


---

## 📂 Project Structure

📁 Song-Recommender/

├── app.py # Main Streamlit application

├── Spotify-2000.csv # Dataset containing songs and their features

├── README.md # You're here!

├── requirements.txt # Python dependencies


---

## 📌 Features

- 🎵 Recommend similar songs using deep learning (autoencoder)
- 💡 Custom styling for a modern dark-themed UI
- 🧠 Learns hidden patterns from 2000 Spotify tracks
- 🎛 Adjustable number of recommendations
- 📈 Embeddings visualized using cosine similarity
- ⚡ Real-time interaction powered by Streamlit

---

## 🧠 How It Works

1. **Dataset**: Contains 2000 songs with metadata and audio features.
2. **Preprocessing**:
   - Missing values are filled
   - Features are normalized using `MinMaxScaler`
3. **Autoencoder**:
   - Learns compressed feature representations (embeddings)
   - Uses 2 dense encoding layers and a decoder
4. **Similarity**:
   - Cosine similarity is calculated between song embeddings
   - Top similar songs are recommended based on user-selected track

---

## 📊 Dataset

- 📁 `Spotify-2000.csv`
- Features used:
  - `Beats Per Minute (BPM)`
  - `Energy`, `Danceability`
  - `Loudness`, `Valence`, `Popularity`, etc.

---

## 📦 Installation & Running Locally


```bash
## 1. Clone the repository
git clone https://github.com/your-username/song-recommender.git
cd song-recommender

## 2. Install dependencies
✅ (Optional but recommended) Create a virtual environment

python -m venv venv

# Activate virtual environment

source venv/bin/activate  # On Windows: venv\Scripts\activate

## 📦 Install required packages

pip install -r requirements.txt

## 3. Run the Streamlit app

streamlit run app.py

```

## 🙋‍♀️ Contributions
Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to modify or improve.
