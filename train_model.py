# ============================================
# TRAIN AI MODEL FROM BOTH DATASETS
# 1. tmdb_5000_movies.csv (International + Turkish + Korean)
# 2. ethiopian_movies.csv (Ethiopian movies)
# ============================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os

print("=" * 70)
print("🎬 TRAINING AI MODEL FROM MULTIPLE DATASETS")
print("=" * 70)

# ==================== LOAD ALL DATASETS ====================
print("\n📊 Step 1: Loading all datasets...")

all_movies = []

# 1. Load TMDB movies (International + Turkish + Korean)
if os.path.exists('tmdb_5000_movies.csv'):
    tmdb_df = pd.read_csv('tmdb_5000_movies.csv', on_bad_lines='skip')
    print(f"   ✅ Loaded {len(tmdb_df)} movies from tmdb_5000_movies.csv")
    all_movies.append(tmdb_df)
else:
    print("   ⚠️ tmdb_5000_movies.csv not found")

# 2. Load Ethiopian movies
if os.path.exists('ethiopian_movies.csv'):
    eth_df = pd.read_csv('ethiopian_movies.csv', on_bad_lines='skip')
    print(f"   ✅ Loaded {len(eth_df)} Ethiopian movies")
    all_movies.append(eth_df)
else:
    print("   ⚠️ ethiopian_movies.csv not found")

# Combine all datasets
if len(all_movies) == 0:
    print("❌ No datasets found! Please provide at least one dataset.")
    exit(1)

df = pd.concat(all_movies, ignore_index=True)
print(f"\n   ✅ TOTAL MOVIES: {len(df)}")

# ==================== DATA PREPROCESSING ====================
print("\n🔧 Step 2: Preprocessing data...")

# Handle missing columns
required_cols = ['title_en', 'genre', 'country', 'year', 'rating', 'description']
for col in required_cols:
    if col not in df.columns:
        if col == 'title_en':
            df['title_en'] = df.get('title', 'Unknown')
        elif col == 'description':
            df['description'] = ''
        elif col == 'rating':
            df['rating'] = 7.0
        elif col == 'year':
            df['year'] = 2020
        else:
            df[col] = 'Unknown'

# Handle Amharic titles
if 'title_am' not in df.columns:
    df['title_am'] = df['title_en']

# Handle missing values
df['description'] = df['description'].fillna('')
df['genre'] = df['genre'].fillna('Unknown')
df['country'] = df['country'].fillna('International')

# Handle optional columns safely
if 'actors' in df.columns:
    df['actors'] = df['actors'].fillna('Unknown')
else:
    df['actors'] = 'Unknown'

if 'director' in df.columns:
    df['director'] = df['director'].fillna('Unknown')
else:
    df['director'] = 'Unknown'

# Create combined features for AI model
df['combined_features'] = (
    df['genre'].astype(str) + ' ' +
    df['country'].astype(str) + ' ' +
    df['description'].astype(str)
)

print("   ✅ Combined features created")

# ==================== ENCODE CATEGORIES ====================
print("\n🏷️ Step 3: Encoding categories...")

le_genre = LabelEncoder()
le_country = LabelEncoder()

df['genre_encoded'] = le_genre.fit_transform(df['genre'])
df['country_encoded'] = le_country.fit_transform(df['country'])

print(f"   ✅ Encoded {len(le_genre.classes_)} genres")
print(f"   ✅ Encoded {len(le_country.classes_)} countries")

# Create popularity score
if 'popularity' not in df.columns:
    if 'vote_average' in df.columns:
        df['popularity'] = df['vote_average'] * 10
    else:
        df['popularity'] = df['rating'] * 10

# ==================== BUILD TF-IDF MODEL ====================
print("\n🧠 Step 4: Building TF-IDF model...")

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"   ✅ TF-IDF Matrix shape: {tfidf_matrix.shape}")
print(f"   ✅ Cosine Similarity shape: {cosine_sim.shape}")

# ==================== SAVE MODELS ====================
print("\n💾 Step 5: Saving models...")

os.makedirs('models', exist_ok=True)

joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
joblib.dump(cosine_sim, 'models/cosine_similarity.pkl')
joblib.dump(le_genre, 'models/label_encoder_genre.pkl')
joblib.dump(le_country, 'models/label_encoder_country.pkl')

# Save metadata
metadata = {
    'num_movies': len(df),
    'num_tmdb': len(tmdb_df) if 'tmdb_df' in locals() else 0,
    'num_ethiopian': len(eth_df) if 'eth_df' in locals() else 0,
    'genres': list(le_genre.classes_),
    'countries': list(le_country.classes_),
    'avg_rating': float(df['rating'].mean()),
    'max_rating': float(df['rating'].max()),
    'top_movie': df.loc[df['rating'].idxmax(), 'title_en'],
    'model_type': 'TF-IDF + Cosine Similarity',
    'trained_date': str(pd.Timestamp.now())
}

with open('models/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

# Save processed dataset
df.to_csv('models/processed_dataset.csv', index=False)

print("   ✅ Models saved to 'models/' folder")
print(f"   📁 Files created:")
print(f"      - models/tfidf_vectorizer.pkl")
print(f"      - models/cosine_similarity.pkl")
print(f"      - models/label_encoder_genre.pkl")
print(f"      - models/label_encoder_country.pkl")
print(f"      - models/metadata.json")
print(f"      - models/processed_dataset.csv")

# ==================== TEST RECOMMENDATIONS ====================
print("\n🎬 Step 6: Testing recommendations...")

def test_recommendation(movie_title):
    try:
        idx = df[df['title_en'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        
        print(f"\n   📍 If you like '{movie_title}':")
        for i, score in sim_scores[:3]:
            movie = df.iloc[i]
            print(f"      → {movie['title_en']} ({movie['genre']}, {movie['country']}) - Score: {score:.3f}")
    except Exception as e:
        print(f"   ❌ '{movie_title}' not found: {e}")

# Test with sample movies
test_movies = ['Inception', 'Parasite', 'Kara Sevda', 'Kara Para Ask', 'Lamb', 'Quraghaye']
for movie in test_movies:
    if movie in df['title_en'].values:
        test_recommendation(movie)
    else:
        print(f"   ❌ '{movie}' not found in dataset")

# ==================== SHOW DATASET STATISTICS ====================
print("\n" + "=" * 70)
print("📊 DATASET STATISTICS")
print("=" * 70)
print(f"\n   Country Distribution:")
for country in df['country'].value_counts().head(15).items():
    print(f"      {country[0]}: {country[1]} movies")

print(f"\n   Genre Distribution:")
for genre in df['genre'].value_counts().head(10).items():
    print(f"      {genre[0]}: {genre[1]} movies")

print(f"\n   Year Range: {df['year'].min()} - {df['year'].max()}")
print(f"   Average Rating: {df['rating'].mean():.2f}/10")

# ==================== SUMMARY ====================
print("\n" + "=" * 70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\n📊 FINAL SUMMARY:")
print(f"   • Total Movies: {len(df)}")
print(f"   • TMDB Movies: {metadata['num_tmdb']}")
print(f"   • Ethiopian Movies: {metadata['num_ethiopian']}")
print(f"   • Genres: {len(le_genre.classes_)}")
print(f"   • Countries: {len(le_country.classes_)}")
print(f"   • Average Rating: {df['rating'].mean():.2f}/10")
print(f"   • Top Rated Movie: {metadata['top_movie']}")
print(f"\n📁 Models saved in 'models/' folder")
print(f"\n🚀 Next: Run 'python app.py' to start the system")
print("=" * 70)