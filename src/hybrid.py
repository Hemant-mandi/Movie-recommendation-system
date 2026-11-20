"""
Hybrid Recommendation Engine.
Combines Content-Based Filtering and Collaborative Filtering (SVD)
using weighted scoring to deliver accurate, personalized, and diverse recommendations.
"""

import pandas as pd
import numpy as np
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender


class HybridRecommender:
    """
    Hybrid Recommender combining Content-Based and Collaborative Filtering.

    Score formulation:
        final_score = alpha * normalized_collab_pred + (1 - alpha) * content_similarity

    Where:
        - alpha: weight given to collaborative filtering (default 0.6)
        - 1 - alpha: weight given to content similarity (default 0.4)
    """

    def __init__(self, alpha=0.6):
        """
        Args:
            alpha: Float between 0.0 and 1.0 balancing CF vs Content-based.
        """
        self.alpha = alpha
        self.content_rec = ContentBasedRecommender()
        self.collab_rec = CollaborativeRecommender()
        self.movies_df = None
        self.ratings_df = None

    def fit(self, movies_df, ratings_df, tags_df=None):
        """Train both models."""
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()

        print("[Hybrid] Training Content-Based Engine...")
        self.content_rec.fit(movies_df, tags_df)

        print("[Hybrid] Training Collaborative Filtering (SVD) Engine...")
        self.collab_rec.fit(ratings_df, movies_df)
        print("[Hybrid] Both engines successfully trained.")

    def recommend(self, user_id, n=10):
        """
        Recommend top-N items for a specific user using hybrid scoring.
        """
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        if len(user_ratings) == 0:
            print(f"User {user_id} has no viewing history. Falling back to popular movies.")
            return self.collab_rec._get_popular_movies(n)

        # 1. Content-based candidates based on user's top-rated movies
        content_candidates = self.content_rec.recommend_for_user(user_id, self.ratings_df, n=50)

        # 2. Collaborative candidates
        collab_candidates = self.collab_rec.recommend(user_id, n=50)

        # Merge candidate sets
        merged_candidates = pd.merge(
            content_candidates[["movieId", "title", "genres", "similarity_score"]],
            collab_candidates[["movieId", "predicted_rating"]],
            on="movieId",
            how="outer"
        )

        # Fill missing values
        # If no similarity score, use median similarity
        med_sim = content_candidates["similarity_score"].median() if len(content_candidates) > 0 else 0.1
        merged_candidates["similarity_score"] = merged_candidates["similarity_score"].fillna(med_sim)

        # If no predicted rating from top list, estimate using SVD model
        def get_pred(row):
            if pd.isna(row["predicted_rating"]):
                return self.collab_rec.predict_rating(user_id, row["movieId"])
            return row["predicted_rating"]

        merged_candidates["predicted_rating"] = merged_candidates.apply(get_pred, axis=1)

        # Normalize predicted ratings (scale 0.5 - 5.0 to 0.0 - 1.0)
        norm_collab = (merged_candidates["predicted_rating"] - 0.5) / 4.5
        norm_collab = norm_collab.clip(0.0, 1.0)

        # Similarity score is already ~ 0.0 to 1.0
        norm_content = merged_candidates["similarity_score"].clip(0.0, 1.0)

        # Calculate hybrid weighted score
        merged_candidates["hybrid_score"] = (
            self.alpha * norm_collab + (1.0 - self.alpha) * norm_content
        )

        # Sort and take top N
        recommendations = merged_candidates.sort_values("hybrid_score", ascending=False).head(n)

        # Format output
        recommendations["predicted_rating"] = recommendations["predicted_rating"].round(2)
        recommendations["similarity_score"] = recommendations["similarity_score"].round(3)
        recommendations["hybrid_score"] = (recommendations["hybrid_score"] * 100).round(1)

        # Retrieve movie metadata if missing from merges
        final_recs = pd.merge(
            recommendations[["movieId", "hybrid_score", "predicted_rating", "similarity_score"]],
            self.movies_df[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        )

        return final_recs[["movieId", "title", "genres", "hybrid_score", "predicted_rating", "similarity_score"]]
