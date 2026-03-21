import streamlit as st
import requests
import pandas as pd
import pickle
import os
import time

@st.cache_data
def load_titles():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    title_path = os.path.join(BASE_DIR, "config", "title.pkl")
    with open(title_path, "rb") as f:
        return pickle.load(f)

titles = load_titles()

# -----------------------------
# CONFIG
# -----------------------------
BACKEND_URL = "http://localhost:8000"
TMDB_API_KEY = "d23ef2eb55299573289d554d461da163"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

# -----------------------------
# FUNCTIONS
# -----------------------------
def fetch_poster(tmdb_id, retry_count=3):
    """Fetch poster with retry logic and rate limiting"""
    if pd.isna(tmdb_id):
        return None

    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
    
    for attempt in range(retry_count):
        try:
            # Add small delay between requests to avoid rate limiting
            if attempt > 0:
                time.sleep(0.5 * attempt)  # Exponential backoff
            
            res = requests.get(url, timeout=5)
            
            if res.status_code == 200:
                poster_path = res.json().get("poster_path")
                if poster_path:
                    return POSTER_BASE + poster_path
                return None
            
            elif res.status_code == 429:  # Too Many Requests
                wait_time = 1 * (attempt + 1)
                time.sleep(wait_time)
                continue
            
            else:
                return None
                
        except requests.exceptions.ConnectionError:
            if attempt < retry_count - 1:
                time.sleep(1)  # Wait before retry
                continue
            return None
        
        except requests.exceptions.Timeout:
            if attempt < retry_count - 1:
                continue
            return None
        
        except Exception as e:
            print(f"Error fetching poster for {tmdb_id}: {e}")
            return None
    
    return None


def get_recommendations(movie, top_k=5):
    """Get recommendations from backend API"""
    try:
        payload = {"movie": movie, "top_k": top_k}
        res = requests.post(f"{BACKEND_URL}/recommend", json=payload, timeout=10)

        if res.status_code != 200:
            st.error(res.json().get("detail", "Unknown error"))
            return []

        return res.json()["recommendations"]
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Is the server running?")
        return []
    
    except requests.exceptions.Timeout:
        st.error("❌ Backend request timed out")
        return []
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return []


# -----------------------------
# UI
# -----------------------------
st.title("🎬 Hybrid Movie Recommender")

movie_name = st.selectbox(
    "Select a movie",
    titles
)

top_k = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend") and movie_name:
    with st.spinner("Getting recommendations..."):
        recs = get_recommendations(movie_name, top_k)

    if recs:
        st.success(f"Found {len(recs)} recommendations!")
        
        # Create columns
        cols = st.columns(min(len(recs), 5))  # Max 5 columns per row
        
        # Fetch posters with progress indicator
        with st.spinner("Loading posters..."):
            for idx, rec in enumerate(recs):
                col_idx = idx % 5
                with cols[col_idx]:
                    # Fetch poster with rate limiting
                    poster = fetch_poster(rec["tmdb_id"])
                    
                    if poster:
                        st.image(poster, use_container_width=True)
                    else:
                        # Placeholder if poster fails
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 60px 20px;
                                text-align: center;
                                border-radius: 8px;
                                color: white;
                                font-weight: bold;
                            ">
                                🎬<br>{rec["title"][:30]}...
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    st.markdown(f"**{rec['title']}**")
                    
                    # Small delay between poster fetches
                    if idx < len(recs) - 1:
                        time.sleep(0.2)
                
                # Start new row after 5 items
                if (idx + 1) % 5 == 0 and idx < len(recs) - 1:
                    cols = st.columns(min(len(recs) - idx - 1, 5))
    else:
        st.warning("No recommendations found")

# Footer
st.markdown("---")
st.caption("Powered by TMDB API • Backend: FastAPI + Sparse Matrices")