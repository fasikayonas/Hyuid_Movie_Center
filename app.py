# ============================================================
import os
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, render_template_string, request, jsonify, session, redirect
import requests
import json
import os
import re
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = "debre_berhan_university_2024"

# ================= TMDB CONFIG =================
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

# ================= USER DATABASE =================
USER_FILE = "users.json"

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(data):
    with open(USER_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ================= VALIDATION =================
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    return len(password) >= 4

# ================= FETCH MOVIE DETAILS FROM TMDB =================
def fetch_movie_details(title):
    try:
        search_url = f"{BASE_URL}/search/movie"
        res = requests.get(search_url, params={
            "api_key": API_KEY,
            "query": title
        }, timeout=5)
        
        if res.status_code == 200:
            results = res.json().get("results", [])
            if results:
                movie = results[0]
                movie_id = movie.get("id")
                poster_path = movie.get("poster_path")
                overview = movie.get("overview", "No description available")
                rating = movie.get("vote_average", 0)
                
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""
                
                trailer_url = ""
                if movie_id:
                    video_url = f"{BASE_URL}/movie/{movie_id}/videos"
                    video_res = requests.get(video_url, params={"api_key": API_KEY}, timeout=5)
                    if video_res.status_code == 200:
                        for v in video_res.json().get("results", []):
                            if v["type"] == "Trailer" and v["site"] == "YouTube":
                                trailer_url = f"https://www.youtube.com/watch?v={v['key']}"
                                break
                
                return poster_url, overview, rating, trailer_url
        return "", "No description available", 0, ""
    except:
        return "", "No description available", 0, ""

# ================= LOAD TRAINED MODEL =================
# ================= LOAD TRAINED MODEL =================
print("=" * 50)
print("🎬 LOADING TRAINED AI MODEL...")
print("=" * 50)

movies_data = []
TRAINED_MODEL_LOADED = False

try:
    if os.path.exists('models/processed_dataset.csv'):
        df_model = pd.read_csv('models/processed_dataset.csv')
        similarity_model = joblib.load('models/cosine_similarity.pkl')
        
        print(f"✅ TRAINED MODEL LOADED!")
        print(f"   📊 Total Movies: {len(df_model)}")
        print(f"   🌍 Countries: {df_model['country'].nunique()}")
        print(f"   ⭐ Average Rating: {df_model['rating'].mean():.2f}/10")
        print(f"\n📸 Fetching movie details from TMDB...")
        
        for idx, row in df_model.iterrows():
            poster_url, overview, rating, trailer_url = fetch_movie_details(row['title_en'])
            
            movies_data.append({
                "id": idx,
                "title": row['title_en'],
                "overview": overview if overview else str(row.get('description', 'No description')),
                "poster": poster_url,
                "rating": rating if rating > 0 else float(row.get('rating', 7.0)),
                "year": str(row.get('year', '2020'))[:4],
                "youtube": trailer_url,
                "country": str(row.get('country', 'International')),
                "genre": str(row.get('genre', 'Unknown'))
            })
            
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx+1}/{len(df_model)} movies...")
        
        print(f"   ✅ Processed {len(movies_data)} movies!")
        TRAINED_MODEL_LOADED = True
        
        def recommend(movie_title):
            """Returns searched movie FIRST, then similar movies from same genre"""
            search_term = movie_title.lower().strip()
            
            idx = None
            target_genre = None
            searched_movie = None
            
            # Find the movie
            for i, row in df_model.iterrows():
                db_title = str(row.get('title_en', '')).lower().strip()
                if search_term == db_title or search_term in db_title or db_title in search_term:
                    idx = i
                    target_genre = row.get('genre', 'Unknown')
                    searched_movie = row.get('title_en', '')
                    print(f"Found: {searched_movie} (Genre: {target_genre})")
                    break
            
            if idx is None:
                print(f"Movie '{movie_title}' not found")
                return []
            
            # Get similarity scores
            scores = list(enumerate(similarity_model[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            results = []
            seen_titles = set()
            
            # 1. FIRST - Add the searched movie itself
            searched_movie_data = next((m for m in movies_data if m['title'].lower() == searched_movie.lower()), None)
            if searched_movie_data:
                results.append(searched_movie_data)
                seen_titles.add(searched_movie_data['title'].lower())
            
            # 2. THEN - Add similar movies from same genre
            for i, score in scores[1:30]:
                if len(results) >= 6:
                    break
                row = df_model.iloc[i]
                movie_genre = row.get('genre', 'Unknown')
                movie_title_db = row.get('title_en', '')
                
                if movie_genre == target_genre and movie_title_db.lower() not in seen_titles:
                    movie_details = next((m for m in movies_data if m['title'].lower() == movie_title_db.lower()), None)
                    if movie_details:
                        results.append(movie_details)
                        seen_titles.add(movie_title_db.lower())
            
            # If no same-genre movies found, add top similar regardless of genre
            if len(results) <= 1:
                for i, score in scores[1:10]:
                    if len(results) >= 6:
                        break
                    row = df_model.iloc[i]
                    movie_title_db = row.get('title_en', '')
                    if movie_title_db.lower() not in seen_titles:
                        movie_details = next((m for m in movies_data if m['title'].lower() == movie_title_db.lower()), None)
                        if movie_details:
                            results.append(movie_details)
                            seen_titles.add(movie_title_db.lower())
            
            return results
    
    else:
        print("⚠️ Trained model not found. Run train_model.py first!")
        TRAINED_MODEL_LOADED = False
        
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    TRAINED_MODEL_LOADED = False

# ================= CHATBOT =================
def chatbot_response(user_text):
    text = user_text.lower()
    
    # Greetings
    if text in ["hi", "hello", "hey", "selam"]:
        return "👋 Hello! I am Hyuid AI Movie Bot. Ask me about movies from Ethiopia 🇪🇹, Turkey 🇹🇷, Korea 🇰🇷, India 🇮🇳, or Hollywood 🇺🇸!"
    
    # Ethiopian Movies
    elif "ethiopian" in text or "amharic" in text:
        return "🇪🇹 Ethiopian movies: Lamb (ላምብ), Quraghaye (ቁራጋዬ), Sriyet (ሥርየት), Ye Wendoch Guday (የወንዶች ጉዳይ), 6 Seat Ke Lelitu, Tikur Aday, Doka, Difret, Ensaro, Anlakekm"
    
    # Turkish Series
    elif "turkish" in text:
        return "🇹🇷 Turkish series: Kuruluş Osman, Golden Boy (Yalı Çapkını), The Pit (Çukur), Family Secrets (Yargı), Love Is in the Air (Sen Çal Kapımı), Fatmagul, Ask Memnu, Karasevda"
    
    # Korean
    elif "korean" in text:
        return "🇰🇷 Korean content: Parasite (Movie), Squid Game (Series), Perfect Crown (Series), Crash Landing On You, True Beauty"
    
    # Indian
    elif "indian" in text or "bollywood" in text:
        return "🇮🇳 Indian movies: 3 Idiots, Dangal, RRR, Bahubali, PK"
    
    # Action
    elif "action" in text:
        return "🔥 Action movies: Inception, Joker, The Dark Knight, Gladiator, John Wick, Ensaro"
    
    # Romance
    elif "romance" in text:
        return "❤️ Romance movies: Titanic, The Notebook, Ask Memnu, Love Is in the Air, Taza"
    
    # Comedy
    elif "comedy" in text:
        return "😂 Comedy movies: 3 Idiots, Ye Wendoch Guday, The Hangover, Qemis"
    
    # Animated
    elif "animated" in text or "animation" in text:
        return "🎨 Animated movies: Lion King, Frozen, Toy Story, Coco, Shrek, Spider-Verse"
    
    # How to get recommendations
    elif "recommend" in text:
        return "🎯 Type a movie name in the 'AI Recommendation' box! You'll see the movie + similar movies from same genre."
    
    # Help
    elif "help" in text:
        return "💡 Ask me about: Ethiopian 🇪🇹, Turkish 🇹🇷, Korean 🇰🇷, Indian 🇮🇳, Action, Romance, Comedy, or Animated movies!"
    elif "who made you" in text or "who created" in text or "developer" in text:
        return "🤖 I was created by Group 6 students using Python, Flask, and Machine Learning !"
    # Bye
    elif "bye" in text:
        return "👋 Goodbye! Enjoy watching movies!"
    
    else:
        return "🤖 Ask me about Ethiopian, Turkish, Korean, Indian, Action, Romance, Comedy, or Animated movies! Type 'help' for options."

# ================= HTML TEMPLATE =================
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Hyuid Movie AI</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        .goog-te-banner-frame.skiptranslate { display: none !important; }
        body { top: 0px !important; font-family: Arial, sans-serif; min-height: 100vh; background: url('https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=2070') no-repeat center center fixed; background-size: cover; }
        body::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.75); z-index: -1; }
        .google-translate { position: fixed; top: 15px; right: 15px; z-index: 1000; background: rgba(0,0,0,0.6); padding: 4px 8px; border-radius: 25px; }
        .google-translate .goog-te-combo { background: #e94560; color: white; border: none; border-radius: 20px; padding: 5px 12px; cursor: pointer; }
        .header { padding: 20px; background: rgba(0,0,0,0.5); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; }
        .header h2 { color: #e94560; font-size: 24px; }
        .header-right { display: flex; gap: 15px; align-items: center; }
        .header-right button { padding: 8px 15px; border-radius: 8px; border: none; cursor: pointer; background: #e94560; color: white; font-weight: bold; }
        .stats { background: rgba(0,0,0,0.7); padding: 5px 12px; border-radius: 20px; color: #e94560; font-size: 12px; }
        .card { display: inline-block; background: rgba(255,255,255,0.95); margin: 10px; padding: 10px; width: 200px; border-radius: 10px; cursor: pointer; transition: transform 0.3s; color: #333; vertical-align: top; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 5px 20px rgba(0,0,0,0.3); }
        .card img { width: 100%; border-radius: 10px; height: 280px; object-fit: cover; }
        .card .no-poster { height: 280px; background: #333; display: flex; align-items: center; justify-content: center; color: white; border-radius: 10px; }
        input { padding: 10px; margin: 5px; border: none; border-radius: 8px; width: 250px; }
        .section { padding: 20px; }
        .section h3 { color: white; margin-bottom: 15px; }
        button { padding: 10px 20px; background: #e94560; color: white; border: none; border-radius: 8px; cursor: pointer; }
        .country-tag { display: inline-block; background: #e94560; color: white; font-size: 10px; padding: 2px 6px; border-radius: 10px; margin-top: 5px; }
        .genre-tag { display: inline-block; background: #4CAF50; color: white; font-size: 10px; padding: 2px 6px; border-radius: 10px; margin-top: 5px; margin-left: 5px; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.95); z-index: 2000; overflow-y: auto; }
        .modal-content { max-width: 900px; margin: 50px auto; background: white; border-radius: 20px; padding: 30px; color: #333; }
        .close { float: right; font-size: 28px; cursor: pointer; color: #999; }
        .close:hover { color: #e94560; }
        .movie-player { width: 100%; height: 400px; margin: 15px 0; }
        .movie-player iframe { width: 100%; height: 100%; border-radius: 10px; }
        .back-button { background: #6c757d; color: white; padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; margin-bottom: 20px; }
        .error-message { color: #e94560; font-size: 12px; margin-top: 5px; display: none; }
        @media (max-width: 768px) { .card { width: 160px; } .card img, .card .no-poster { height: 220px; } .movie-player { height: 250px; } }
    </style>
    <script type="text/javascript">
        function googleTranslateElementInit() {
            new google.translate.TranslateElement({
                pageLanguage: 'en',
                includedLanguages: 'am,en',
                layout: google.translate.TranslateElement.InlineLayout.SIMPLE
            }, 'google_translate_element');
        }
    </script>
    <script src="//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
</head>
<body>

<div class="google-translate" id="google_translate_element"></div>

<div id="loginContainer" style="display: flex; justify-content: center; align-items: center; min-height: 100vh;">
    <div style="background: rgba(0,0,0,0.85); padding: 40px; border-radius: 20px; text-align: center; width: 380px; backdrop-filter: blur(10px); border: 1px solid #e94560;">
        <h2 style="color:#e94560;">🎬 Hyuid Movie Center</h2>
       
        <input type="email" id="email" placeholder="Email" style="width:100%; margin:10px 0; padding:12px; border-radius:8px; border:none;">
        <div id="emailError" class="error-message">❌ Invalid email format!</div>
        <input type="password" id="password" placeholder="Password (min 4 characters)" style="width:100%; margin:10px 0; padding:12px; border-radius:8px; border:none;">
        <div id="passwordError" class="error-message">❌ Password must be at least 4 characters!</div>
        <button onclick="doLogin()" style="width:100%; padding:12px; background:#e94560; color:white; border:none; border-radius:8px; cursor:pointer;">🔐 Login / Register</button>
    </div>
</div>

<div id="mainApp" style="display: none;">
    <div class="header">
        <h2>🎬 Hyuid Movie Center</h2>
        <div class="header-right">
            <span class="stats">🎯 {{ movies_count }} Movies | 11 Countries</span>
            <span id="userName" style="color:white;"></span>
            <button onclick="doLogout()">🚪 Logout</button>
        </div>
    </div>
    <div class="section">
        <h3>🔥 Trending Movies</h3>
        <div id="movies"></div>
    </div>
    <div class="section">
        <h3>🤖 AI Recommendation</h3>
        <p style="color:#aaa; font-size:12px;">💡 You will see the movie you searched + similar movies from the same genre!</p>
        <input id="movie" placeholder="Enter movie title (e.g., Inception, Lamb, Parasite)">
        <button onclick="getRec()">Get Recommendations</button>
        <div id="rec" style="margin-top:15px;"></div>
    </div>
    <div class="section">
        <h3>💬 AI Movie Chatbot</h3>
        <input id="chatInput" placeholder="Ask about Ethiopian, Turkish, Korean, Indian movies...">
        <button onclick="sendChat()">Send</button>
        <div id="chatBox" style="margin-top:10px; color:white; max-height:300px; overflow-y:auto;"></div>
    </div>
</div>

<div id="movieModal" class="modal">
    <div class="modal-content">
        <span class="close" onclick="closeModal()">&times;</span>
        <button class="back-button" onclick="goBackToMovies()">← Back to Movies</button>
        <div id="modalBody"></div>
    </div>
</div>

<script>
    let moviesData = {{ movies|safe }};
    
    console.log("Movies loaded:", moviesData.length);
    
    function doLogin() {
        let email = document.getElementById('email').value;
        let password = document.getElementById('password').value;
        let isValid = true;
        
        document.getElementById('emailError').style.display = 'none';
        document.getElementById('passwordError').style.display = 'none';
        
        if (!email.includes('@') || !email.includes('.')) {
            document.getElementById('emailError').style.display = 'block';
            isValid = false;
        }
        if (password.length < 4) {
            document.getElementById('passwordError').style.display = 'block';
            isValid = false;
        }
        if (!isValid) return;
        
        fetch("/do_login", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({email: email, password: password})
        })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                localStorage.setItem('userEmail', email);
                document.getElementById('loginContainer').style.display = 'none';
                document.getElementById('mainApp').style.display = 'block';
                document.getElementById('userName').innerHTML = `👋 Welcome ${email.split('@')[0]}!`;
                showMovies();
            } else {
                alert(data.message);
            }
        });
    }
    
    function doLogout() {
        localStorage.removeItem('userEmail');
         document.getElementById("chatBox").innerHTML = "";
           document.getElementById("rec").innerHTML = "";
             document.getElementById("movie").value = "";
    document.getElementById("chatInput").value = "";
        document.getElementById('loginContainer').style.display = 'flex';
        document.getElementById('mainApp').style.display = 'none';
        document.getElementById('email').value = '';
        document.getElementById('password').value = '';
         
    }
    
    function goBackToMovies() {
        closeModal();
    }
    
    function showMovies() {
        let html = "";
        if (moviesData.length === 0) {
            html = "<p style='color:white;'>No movies found. Please run train_model.py first!</p>";
        } else {
            moviesData.slice(0, 24).forEach(m => {
                let posterHtml = m.poster ? `<img src="${m.poster}" onerror="this.src=''">` : `<div class="no-poster">🎬<br>${m.title}</div>`;
                html += `<div class="card" onclick="openMovie(${m.id})">
                    ${posterHtml}
                    <h4>${m.title.substring(0, 25)}</h4>
                    <p>⭐ ${typeof m.rating === 'number' ? m.rating.toFixed(1) : m.rating}</p>
                    <p>📅 ${m.year}</p>
                    ${m.country ? `<span class="country-tag">${m.country}</span>` : ''}
                    ${m.genre ? `<span class="genre-tag">${m.genre}</span>` : ''}
                </div>`;
            });
        }
        document.getElementById("movies").innerHTML = html;
    }
    
    function getRec() {
    let title = document.getElementById("movie").value;
    if (!title) { alert('Enter a movie title!'); return; }
    
    fetch("/recommend", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({title: title})
    })
    .then(r => r.json())
    .then(data => {
        let html = "<h4 style='color:white;'>🎯 Recommendations (Searched Movie + Similar Movies):</h4><div style='display:flex; flex-wrap:wrap;'>";
        if (!data || data.length === 0) {
            html += "<p style='color:white;'>No recommendations found. Try: Inception, Parasite, Lamb, or 3 Idiots</p>";
        } else {
            data.forEach(m => {
                let posterHtml = m.poster ? `<img src="${m.poster}" style="height:200px; object-fit:cover; width:100%; border-radius:10px;" onerror="this.src=''">` : `<div style="height:200px; background:#333; display:flex; align-items:center; justify-content:center; color:white; border-radius:10px;">🎬<br>${m.title.substring(0,15)}</div>`;
                html += `<div style="width:180px; margin:10px; cursor:pointer;" onclick="openMovie(${m.id})">
                    ${posterHtml}
                    <p style="color:white; margin-top:5px;"><strong>${m.title.substring(0, 25)}</strong><br>⭐ ${typeof m.rating === 'number' ? m.rating.toFixed(1) : m.rating}<br><span style="font-size:10px; color:#4CAF50;">${m.genre || ''}</span></p>
                </div>`;
            });
        }
        html += "</div>";
        document.getElementById("rec").innerHTML = html;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("rec").innerHTML = "<p style='color:white;'>Error getting recommendations. Please try again.</p>";
    });
}
    
   function openMovie(movieId) {
    let movie = moviesData.find(m => m.id == movieId);
    if (!movie) return;
    
    // Fetch recommendations for this movie
    fetch("/recommend", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({title: movie.title})
    })
    .then(r => r.json())
    .then(recommendations => {
        let recHtml = "";
        if (recommendations && recommendations.length > 0) {
            recHtml = "<h3 style='margin-top:20px;'>🎯 Similar Movies (Same Genre):</h3><div style='display:flex; flex-wrap:wrap;'>";
            recommendations.forEach(rec => {
                // Don't show the same movie again
                if (rec.id != movieId) {
                    let recPoster = rec.poster ? `<img src="${rec.poster}" style="width:100%; height:180px; object-fit:cover; border-radius:10px;" onerror="this.src=''">` : `<div style="width:100%; height:180px; background:#333; display:flex; align-items:center; justify-content:center; color:white; border-radius:10px;">🎬</div>`;
                    recHtml += `
                        <div style="width:150px; margin:10px; text-align:center; cursor:pointer;" onclick="openMovie(${rec.id})">
                            ${recPoster}
                            <p style="margin-top:5px;"><strong>${rec.title.substring(0, 20)}</strong><br>⭐ ${rec.rating.toFixed(1)}<br><span style="font-size:10px; color:#4CAF50;">${rec.genre || ''}</span></p>
                        </div>
                    `;
                }
            });
            recHtml += "</div>";
        } else {
            recHtml = "<p style='margin-top:20px; color:#999;'>No similar movies found.</p>";
        }
        
        let playerHtml = "";
        if (movie.youtube && movie.youtube.includes('v=')) {
            let videoId = movie.youtube.split('v=')[1];
            playerHtml = `<div class="movie-player"><iframe src="https://www.youtube.com/embed/${videoId}" frameborder="0" allowfullscreen></iframe></div>`;
        } else {
            playerHtml = `<p style="color:#999;">🎬 No trailer available for this movie</p>`;
        }
        
        document.getElementById("modalBody").innerHTML = `
            <div style="display:flex; gap:20px; flex-wrap:wrap;">
                <img src="${movie.poster}" style="width:200px; border-radius:10px;" onerror="this.src=''">
                <div style="flex:1;">
                    <h2>🎬 ${movie.title}</h2>
                    <p><strong>⭐ Rating:</strong> ${movie.rating.toFixed(1)}/10</p>
                    <p><strong>📅 Year:</strong> ${movie.year}</p>
                    <p><strong>🌍 Country:</strong> ${movie.country || 'International'}</p>
                    <p><strong>🎭 Genre:</strong> <span style="background:#4CAF50; color:white; padding:3px 8px; border-radius:10px;">${movie.genre || 'Unknown'}</span></p>
                    <p><strong>📖 Description:</strong> ${movie.overview || 'No description available'}</p>
                    ${playerHtml}
                </div>
            </div>
            ${recHtml}
            <div style="margin-top:20px; text-align:center;">
                <button class="back-button" onclick="closeModal()">← Close</button>
            </div>
        `;
        document.getElementById("movieModal").style.display = "block";
    });
}
    
    function closeModal() {
        document.getElementById("movieModal").style.display = "none";
    }
    
    function sendChat() {
        let msg = document.getElementById("chatInput").value;
        if (!msg) return;
        document.getElementById("chatBox").innerHTML += `<p><b>You:</b> ${msg}</p>`;
        document.getElementById("chatInput").value = "";
        fetch("/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({message: msg})})
        .then(r => r.json())
        .then(data => {
            document.getElementById("chatBox").innerHTML += `<p><b>🤖 Bot:</b> ${data.reply}</p><hr>`;
            document.getElementById("chatBox").scrollTop = document.getElementById("chatBox").scrollHeight;
        });
    }
    
    let savedEmail = localStorage.getItem('userEmail');
    if (savedEmail) {
        document.getElementById('loginContainer').style.display = 'none';
        document.getElementById('mainApp').style.display = 'block';
        document.getElementById('userName').innerHTML = `👋 Welcome ${savedEmail.split('@')[0]}!`;
        showMovies();
    }
</script>
</body>
</html>
"""

# ================= ROUTES =================
@app.route("/")
def index():
    return redirect("/home")

@app.route("/home")
def home():
    return render_template_string(HTML, movies=movies_data, movies_count=len(movies_data))

@app.route("/do_login", methods=["POST"])
def do_login():
    data = request.json
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    
    if not validate_email(email):
        return jsonify({"success": False, "message": "Invalid email format!"})
    
    if not validate_password(password):
        return jsonify({"success": False, "message": "Password must be at least 4 characters!"})
    
    users = load_users()
    
    if email not in users:
        users[email] = {"password": password, "preference": "popular"}
        save_users(users)
        session["user"] = email
        return jsonify({"success": True, "message": "Registration successful!"})
    
    if users[email]["password"] == password:
        session["user"] = email
        return jsonify({"success": True, "message": "Login successful!"})
    
    return jsonify({"success": False, "message": "Incorrect password!"})

@app.route("/recommend", methods=["POST"])
def recommend_api():
    data = request.json
    title = data.get("title", "")
    return jsonify(recommend(title))

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    response = chatbot_response(user_message)
    return jsonify({"reply": response})

@app.route("/me")
def me():
    return jsonify(session.get("user", None))

# ================= RUN =================
if __name__ == "__main__":
    print("=" * 50)
    print("🎬 HYAID MOVIE CENTER - RUNNING")
    print("=" * 50)
    print("📍 http://localhost:5000")
    if TRAINED_MODEL_LOADED:
        print(f"✅ USING TRAINED MODEL: {len(movies_data)} movies")
        print(f"   First few movies: {[m['title'] for m in movies_data[:5]]}")
    else:
        print("⚠️ No trained model loaded!")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)