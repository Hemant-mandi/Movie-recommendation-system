"""
StreamPulse — OTT Recommendation & Viewing Analytics Platform
Subject: Data Analytics (BCS714D)
A modern, creative, human-crafted OTT streaming interface with personalized recommendations.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure project root is accessible
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils import load_movies, load_ratings, load_tags, clean_data, get_movie_stats
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender

# --- Page Configuration ---
st.set_page_config(
    page_title="StreamPulse | OTT Recommendation System",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling: Sleek Dark OTT Aesthetic ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Background accent adjustments */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #171923 0%, #0d0f17 90%);
        color: #e2e8f0;
    }

    /* Custom Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #2a0808 0%, #151824 60%, #111420 100%);
        padding: 30px 36px;
        border-radius: 18px;
        border: 1px solid rgba(229, 9, 20, 0.25);
        margin-bottom: 28px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 8px;
    }
    .hero-badge {
        background: rgba(229, 9, 20, 0.15);
        color: #ff4d4d;
        border: 1px solid #ff4d4d44;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-top: 12px;
    }

    /* Movie Cards */
    .movie-card {
        background: #181b26;
        border: 1px solid #272d3f;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .movie-card:hover {
        transform: translateY(-4px);
        border-color: #e50914;
        box-shadow: 0 8px 24px rgba(229, 9, 20, 0.2);
    }
    .match-pill {
        background: linear-gradient(135deg, #e50914, #b20710);
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .rating-pill {
        background: #2a2215;
        color: #facc15;
        border: 1px solid #facc1544;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .card-title {
        color: #ffffff;
        font-size: 17px;
        font-weight: 600;
        margin: 12px 0 8px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .genre-chip {
        display: inline-block;
        background: #232838;
        color: #94a3b8;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 4px;
        margin-right: 5px;
        margin-bottom: 4px;
    }
    .stat-box {
        background: #181c2b;
        border: 1px solid #283049;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .stat-number {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    .stat-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Cached Data & Models ---
@st.cache_data
def get_dataset():
    movies = load_movies("small")
    ratings = load_ratings("small")
    tags = load_tags("small")
    movies_clean, ratings_clean, merged = clean_data(movies, ratings)
    return movies_clean, ratings_clean, tags, merged


@st.cache_resource
def get_trained_models(_movies, _ratings, _tags):
    cb = ContentBasedRecommender()
    cb.fit(_movies, _tags)

    cf = CollaborativeRecommender(n_factors=30, n_epochs=15)
    cf.fit(_ratings, _movies)

    hybrid = HybridRecommender(alpha=0.65)
    hybrid.content_rec = cb
    hybrid.collab_rec = cf
    hybrid.movies_df = _movies
    hybrid.ratings_df = _ratings

    return cb, cf, hybrid


movies, ratings, tags, merged = get_dataset()
cb_model, cf_model, hybrid_model = get_trained_models(movies, ratings, tags)

# --- Sidebar: User Persona & Navigation ---
with st.sidebar:
    st.markdown("### 🍿 **StreamPulse**")
    st.caption("OTT Analytics & Recommendation Engine")
    st.markdown("---")

    selected_view = st.radio(
        "Explore Platform",
        [
            "🔥 Trending & Insights",
            "🎬 Watch Next (Content Similarity)",
            "👤 Just For You (User Collaborative)",
            "🎛️ Taste Tuner (Hybrid Engine)"
        ]
    )

    st.markdown("---")
    st.markdown("**Academic Context**")
    st.caption("Course: Data Analytics (BCS714D)\nTech: Hadoop MapReduce + SVD")


# --- Top Hero Banner ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🍿 StreamPulse Recommender</div>
    <div class="hero-subtitle">Personalized movie and web series recommendation system analyzing viewer engagements, rating behaviors, and content metadata.</div>
    <div class="hero-badge">Big Data Analytics • BCS714D</div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# VIEW 1: TRENDING & PLATFORM INSIGHTS
# ==============================================================================
if selected_view == "🔥 Trending & Insights":
    st.subheader("Platform At A Glance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{len(movies):,}</div>
            <div class="stat-label">Catalog Titles</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{len(ratings):,}</div>
            <div class="stat-label">User Reviews</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{ratings['userId'].nunique():,}</div>
            <div class="stat-label">Active Viewers</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{ratings['rating'].mean():.2f} ★</div>
            <div class="stat-label">Mean Platform Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Visual Analytics
    g1, g2 = st.columns([1, 1])

    with g1:
        st.markdown("##### 📈 What are users rating?")
        rating_counts = ratings["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Stars", "Reviews"]
        fig_r = px.bar(
            rating_counts,
            x="Stars",
            y="Reviews",
            color="Reviews",
            color_continuous_scale=["#334155", "#e50914"],
            template="plotly_dark"
        )
        fig_r.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with g2:
        st.markdown("##### 🎭 Top Streaming Genres")
        genres_exploded = movies["genres"].str.split("|").explode().value_counts().head(8).reset_index()
        genres_exploded.columns = ["Genre", "Total Titles"]
        fig_g = px.pie(
            genres_exploded,
            names="Genre",
            values="Total Titles",
            hole=0.55,
            color_discrete_sequence=px.colors.sequential.RdBu,
            template="plotly_dark"
        )
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("##### 🔥 Most Watched Movies Across All Audiences")
    top_stats = get_movie_stats(merged).head(6)

    cols = st.columns(3)
    for idx, (_, row) in enumerate(top_stats.iterrows()):
        col = cols[idx % 3]
        genres = " ".join([f"<span class='genre-chip'>{g}</span>" for g in row['genres'].split('|')[:3]])
        with col:
            st.markdown(f"""
            <div class="movie-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="match-badge" style="background:#232838; color:#94a3b8; font-size:11px; padding:2px 8px; border-radius:4px;">#{idx+1} POPULAR</span>
                    <span class="rating-pill">★ {row['avg_rating']}</span>
                </div>
                <div class="card-title" title="{row['title']}">{row['title']}</div>
                <div>{genres}</div>
                <div style="color:#64748b; font-size:12px; margin-top:8px;">👥 {row['num_ratings']:,} reviews logged</div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: CONTENT SIMILARITY (WATCH NEXT)
# ==============================================================================
elif selected_view == "🎬 Watch Next (Content Similarity)":
    st.subheader("Because You Watched...")
    st.caption("Computes TF-IDF vector similarity over genre tags, metadata tokens, and viewer tags to recommend related stories.")

    all_titles = sorted(movies["title"].unique())

    # Curated popular defaults for quick testing
    popular_samples = ["Inception (2010)", "Toy Story (1995)", "Dark Knight, The (2008)", "Matrix, The (1999)", "Pulp Fiction (1994)"]
    valid_defaults = [t for t in popular_samples if t in all_titles]

    c1, c2 = st.columns([3, 1])
    with c1:
        chosen_title = st.selectbox(
            "Pick a movie you love:",
            all_titles,
            index=all_titles.index(valid_defaults[0]) if valid_defaults else 0
        )
    with c2:
        top_k = st.slider("Suggestions count", min_value=3, max_value=9, value=6, step=3)

    if st.button("Find Similar Titles", type="primary"):
        with st.spinner("Analyzing storytelling style, themes, and genres..."):
            recs = cb_model.recommend(chosen_title, n=top_k)

            if not recs.empty:
                st.write("")
                cols = st.columns(3)
                for idx, (_, row) in enumerate(recs.iterrows()):
                    col = cols[idx % 3]
                    similarity_pct = int(row["similarity_score"] * 100)
                    genres = " ".join([f"<span class='genre-chip'>{g}</span>" for g in row['genres'].split('|')[:3]])
                    with col:
                        st.markdown(f"""
                        <div class="movie-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="match-pill">{similarity_pct}% MATCH</span>
                                <span style="font-size:11px; color:#64748b;">CONTENT-BASED</span>
                            </div>
                            <div class="card-title" title="{row['title']}">{row['title']}</div>
                            <div>{genres}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No direct matches found. Try another movie title.")


# ==============================================================================
# VIEW 3: COLLABORATIVE FILTERING (FOR YOU)
# ==============================================================================
elif selected_view == "👤 Just For You (User Collaborative)":
    st.subheader("Personalized Viewer Profile")
    st.caption("Powered by Singular Value Decomposition (SVD) matrix factorization to predict ratings for movies you haven't seen.")

    available_users = sorted(ratings["userId"].unique())
    user_col, count_col = st.columns([3, 1])
    with user_col:
        selected_user = st.selectbox(
            "Select Viewer Profile ID:",
            available_users,
            index=0,
            help="Each user has their own unique rating history."
        )
    with count_col:
        num_recs = st.slider("Recommendations count", min_value=3, max_value=9, value=6, step=3)

    user_history = merged[merged["userId"] == selected_user].sort_values("rating", ascending=False).head(4)

    st.markdown(f"##### 📜 What Viewer #{selected_user} recently loved:")
    hist_cols = st.columns(4)
    for idx, (_, h_row) in enumerate(user_history.iterrows()):
        with hist_cols[idx]:
            st.markdown(f"""
            <div style="background:#131620; border:1px solid #1f2433; border-radius:10px; padding:12px;">
                <span class="rating-pill">★ {h_row['rating']}</span>
                <div style="color:#ffffff; font-size:13px; font-weight:600; margin-top:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{h_row['title']}</div>
                <div style="color:#64748b; font-size:11px;">{h_row['genres'].split('|')[0]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    if st.button("Generate Personalized Recs", type="primary"):
        with st.spinner("Decomposing latent user-item matrix..."):
            user_recs = cf_model.recommend(user_id=selected_user, n=num_recs)

            st.markdown(f"##### 🎯 Recommended For Viewer #{selected_user}:")
            cols = st.columns(3)
            for idx, (_, row) in enumerate(user_recs.iterrows()):
                col = cols[idx % 3]
                genres = " ".join([f"<span class='genre-chip'>{g}</span>" for g in row['genres'].split('|')[:3]])
                with col:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="match-pill">SVD PREDICTION</span>
                            <span class="rating-pill">★ {row['predicted_rating']}</span>
                        </div>
                        <div class="card-title" title="{row['title']}">{row['title']}</div>
                        <div>{genres}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ==============================================================================
# VIEW 4: TASTE TUNER (HYBRID RECOMMENDER)
# ==============================================================================
elif selected_view == "🎛️ Taste Tuner (Hybrid Engine)":
    st.subheader("Hybrid Recommendation Tuner")
    st.caption("Fine-tune how much you want to rely on collaborative rating history versus matching movie themes and genres.")

    t_user_col, t_slider_col = st.columns([1, 2])
    with t_user_col:
        tuner_user = st.selectbox("Target Viewer:", sorted(ratings["userId"].unique()), index=0)
    with t_slider_col:
        alpha_val = st.slider(
            "Balance Dial (Collaborative Taste vs Content Genre)",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.05,
            help="0.0 = 100% Genre Matching, 1.0 = 100% Collaborative Rating Prediction"
        )

    st.markdown(f"""
    <div style="display:flex; gap:16px; margin: 12px 0 24px 0;">
        <span style="font-size:13px; color:#94a3b8;">🎭 Content Focus: <b>{int((1.0 - alpha_val)*100)}%</b></span>
        <span style="font-size:13px; color:#94a3b8;">👥 Collaborative Taste: <b>{int(alpha_val*100)}%</b></span>
    </div>
    """, unsafe_allow_html=True)

    hybrid_model.alpha = alpha_val

    if st.button("Run Hybrid Engine", type="primary"):
        with st.spinner("Blending content-based vectors with SVD matrices..."):
            h_recs = hybrid_model.recommend(user_id=tuner_user, n=6)

            cols = st.columns(3)
            for idx, (_, row) in enumerate(h_recs.iterrows()):
                col = cols[idx % 3]
                genres = " ".join([f"<span class='genre-chip'>{g}</span>" for g in row['genres'].split('|')[:3]])
                with col:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="match-pill">{row['hybrid_score']}% HYBRID SCORE</span>
                            <span class="rating-pill">★ {row['predicted_rating']}</span>
                        </div>
                        <div class="card-title" title="{row['title']}">{row['title']}</div>
                        <div>{genres}</div>
                        <div style="color:#64748b; font-size:11px; margin-top:8px;">Content Match: {int(row['similarity_score']*100)}%</div>
                    </div>
                    """, unsafe_allow_html=True)
