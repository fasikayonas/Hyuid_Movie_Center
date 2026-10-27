# ============================================
# FETCH 5000+ MOVIES FROM TMDB API
# Run this once to create your dataset
# ============================================

import os
from dotenv import load_dotenv

load_dotenv()

import requests
import pandas as pd
import time

from tqdm import tqdm

print("=" * 60)
print("🎬 FETCHING 5000+ MOVIES FROM TMDB API")
print("=" * 60)

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

def fetch_movies_from_tmdb(limit=5000):
    """Fetch movies directly from TMDB API"""
    all_movies = []
    
    # Fetch from multiple categories
    categories = [
        ("movie/popular", 500),
        ("movie/top_rated", 500),
        ("movie/now_playing", 300),
        ("movie/upcoming", 300),
        ("discover/movie", 4000)  # Discover for more movies
    ]
    
    for category, target_count in categories:
        print(f"\n📡 Fetching from {category}...")
        page = 1
        fetched = 0
        
        while fetched < target_count and page <= 50:
            url = f"{BASE_URL}/{category}"
            params = {
                "api_key": API_KEY,
                "page": page,
                "language": "en-US",
                "sort_by": "popularity.desc"
            }
            
            if category == "discover/movie":
                params["vote_count.gte"] = 100
                params["vote_average.gte"] = 5
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for movie in data.get("results", []):
                    # Get detailed info
                    movie_id = movie["id"]
                    detail_url = f"{BASE_URL}/movie/{movie_id}"
                    detail_res = requests.get(detail_url, params={"api_key": API_KEY})
                    
                    genres = []
                    countries = []
                    youtube_link = ""
                    
                    if detail_res.status_code == 200:
                        detail = detail_res.json()
                        genres = [g["name"] for g in detail.get("genres", [])]
                        countries = [c["name"] for c in detail.get("production_countries", [])]
                        
                        # Get YouTube trailer
                        video_url = f"{BASE_URL}/movie/{movie_id}/videos"
                        video_res = requests.get(video_url, params={"api_key": API_KEY})
                        if video_res.status_code == 200:
                            for v in video_res.json().get("results", []):
                                if v["type"] == "Trailer" and v["site"] == "YouTube":
                                    youtube_link = f"https://www.youtube.com/watch?v={v['key']}"
                                    break
                    
                    all_movies.append({
                        "id": movie_id,
                        "title_en": movie["title"],
                        "title_am": movie["title"],  # Will be translated by Google Translate
                        "genre": genres[0] if genres else "Unknown",
                        "country": countries[0] if countries else "International",
                        "year": int(movie["release_date"][:4]) if movie.get("release_date") else 2020,
                        "rating": round(movie.get("vote_average", 0), 1),
                        "description": movie.get("overview", "No description available"),
                        "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else "",
                        "youtube": youtube_link,
                        "popularity": movie.get("popularity", 0),
                        "votes": movie.get("vote_count", 0)
                    })
                    
                    fetched += 1
                    if fetched >= target_count:
                        break
                
                page += 1
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"   Error on page {page}: {e}")
                break
            
            if len(all_movies) >= limit:
                break
        
        if len(all_movies) >= limit:
            break
    
    return all_movies[:limit]

print("\n⏳ Fetching movies... This may take 5-10 minutes...")
movies = fetch_movies_from_tmdb(5000)
print(f"\n✅ Fetched {len(movies)} movies from TMDB")

# Save to CSV
df = pd.DataFrame(movies)
df.to_csv("tmdb_5000_movies.csv", index=False)
print(f"\n✅ Saved to 'tmdb_5000_movies.csv'")
print(f"   Total movies: {len(df)}")
print(f"   Columns: {list(df.columns)}")