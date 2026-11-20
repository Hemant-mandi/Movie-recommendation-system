"""
Content-Based Recommendation Engine.
Recommends movies based on similarity of genres, tags, and descriptions.
Uses TF-IDF vectorization + cosine similarity.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-based filtering using movie metadata (genres + tags).
    
    How it works:
    1. Combine genres and tags into a single text "soup" per movie
    2. Convert text to TF-IDF vectors
    3. Compute cosine similarity between all movie pairs
    4. For a given movie, return the most similar ones
    """

    def __init__(self):
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.movies_df = None
        self.indices = None  # Maps movie title -> index

    def fit(self, movies_df, tags_df=None):
        """
        Build the content-based model.
        
        Args:
            movies_df: DataFrame with movieId, title, genres
            tags_df: Optional DataFrame with movieId, tag
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)

        # Create a "soup" of features for each movie
        # Start with genres (replace | with space)
        self.movies_df["soup"] = self.movies_df["genres"].str.replace("|", " ", regex=False)

        # Add tags if available
        if tags_df is not None and len(tags_df) > 0:
            # Aggregate all tags per movie into a single string
            movie_tags = tags_df.groupby("movieId")["tag"].apply(
                lambda x: " ".join(x.dropna().astype(str).str.lower())
            ).reset_index()
            movie_tags.columns = ["movieId", "all_tags"]

            self.movies_df = self.movies_df.merge(movie_tags, on="movieId", how="left")
            self.movies_df["all_tags"] = self.movies_df["all_tags"].fillna("")
            self.movies_df["soup"] = self.movies_df["soup"] + " " + self.movies_df["all_tags"]

        # Handle "(no genres listed)" case
        self.movies_df["soup"] = self.movies_df["soup"].str.replace("(no genres listed)", "", regex=False)
        self.movies_df["soup"] = self.movies_df["soup"].str.strip()

        # Build TF-IDF matrix (keep sparse to save memory)
        tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df["soup"])

        # Build title-to-index mapping
        self.indices = pd.Series(
            self.movies_df.index,
            index=self.movies_df["title"].str.lower()
        ).drop_duplicates()

        print(f"Content-based model built: {self.tfidf_matrix.shape[0]} movies, "
              f"{self.tfidf_matrix.shape[1]} features (sparse)")

    def recommend(self, movie_title, n=10):
        """
        Get top-N similar movies based on content.
        Computes cosine similarity on-the-fly for the chosen movie only.
        """
        title_lower = movie_title.lower()

        if title_lower not in self.indices:
            # Try partial match
            matches = self.movies_df[
                self.movies_df["title"].str.lower().str.contains(title_lower, na=False)
            ]
            if len(matches) == 0:
                print(f"Movie '{movie_title}' not found.")
                return pd.DataFrame()
            idx = matches.index[0]
        else:
            idx = self.indices[title_lower]

        # Compute cosine similarity on the fly (1 x N instead of N x N)
        target_vector = self.tfidf_matrix[idx]
        sim_scores_array = cosine_similarity(target_vector, self.tfidf_matrix).flatten()

        # Sort indices by similarity (descending)
        sim_indices = sim_scores_array.argsort()[::-1]

        # Filter out the movie itself
        sim_indices = [i for i in sim_indices if i != idx][:n]

        # Build result DataFrame
        result = self.movies_df.iloc[sim_indices][["movieId", "title", "genres"]].copy()
        result["similarity_score"] = [round(float(sim_scores_array[i]), 4) for i in sim_indices]
        result = result.reset_index(drop=True)

        return result

    def recommend_for_user(self, user_id, ratings_df, n=10):
        """
        Get recommendations for a user based on their highly-rated movies.
        
        Finds movies similar to the user's top-rated ones that they haven't watched.
        
        Args:
            user_id: The user ID
            ratings_df: DataFrame with userId, movieId, rating
            n: Number of recommendations
        
        Returns:
            DataFrame with recommended movies
        """
        # Get user's top-rated movies (rating >= 4.0)
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if len(user_ratings) == 0:
            print(f"User {user_id} not found in ratings.")
            return pd.DataFrame()

        top_movies = user_ratings[user_ratings["rating"] >= 4.0].sort_values(
            "rating", ascending=False
        ).head(5)

        # Get the user's already-watched movie IDs
        watched_ids = set(user_ratings["movieId"].values)

        # Collect recommendations from each top-rated movie
        all_recs = []
        for _, row in top_movies.iterrows():
            movie = self.movies_df[self.movies_df["movieId"] == row["movieId"]]
            if len(movie) > 0:
                recs = self.recommend(movie.iloc[0]["title"], n=20)
                recs = recs[~recs["movieId"].isin(watched_ids)]
                all_recs.append(recs)

        if not all_recs:
            return pd.DataFrame()

        # Combine and deduplicate, keeping highest similarity
        combined = pd.concat(all_recs, ignore_index=True)
        combined = combined.sort_values("similarity_score", ascending=False)
        combined = combined.drop_duplicates(subset="movieId").head(n)

        return combined.reset_index(drop=True)


# ============================================================
# Standalone usage
# ============================================================
if __name__ == "__main__":
    from utils import load_movies, load_tags

    # Load data
    movies = load_movies("small")
    tags = load_tags("small")

    # Build model
    recommender = ContentBasedRecommender()
    recommender.fit(movies, tags)

    # Test: Get recommendations for a movie
    print("\n--- Movies similar to 'Toy Story (1995)' ---")
    recs = recommender.recommend("Toy Story (1995)")
    print(recs.to_string(index=False))
