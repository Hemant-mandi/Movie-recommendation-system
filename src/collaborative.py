"""
Collaborative Filtering Recommendation Engine.
Uses SVD (Singular Value Decomposition) from the Surprise library
to predict user ratings for unseen movies.
"""

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split


class CollaborativeRecommender:
    """
    Collaborative filtering using SVD matrix factorization.
    
    How it works:
    1. Decompose the user-item rating matrix into latent factors
    2. Learn hidden preferences for each user and hidden features for each movie
    3. Predict ratings for movies a user hasn't seen
    4. Recommend top-rated unseen movies
    """

    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """
        Args:
            n_factors: Number of latent factors for SVD
            n_epochs: Number of training iterations
            lr_all: Learning rate
            reg_all: Regularization term
        """
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        self.trainset = None
        self.movies_df = None
        self.ratings_df = None

    def fit(self, ratings_df, movies_df):
        """
        Train the SVD model on ratings data.
        
        Args:
            ratings_df: DataFrame with userId, movieId, rating
            movies_df: DataFrame with movieId, title, genres
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()

        # Prepare data for Surprise library
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings_df[["userId", "movieId", "rating"]],
            reader
        )

        # Build full trainset
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

        print(f"SVD model trained on {self.trainset.n_users} users, "
              f"{self.trainset.n_items} items, "
              f"{self.trainset.n_ratings} ratings")

    def evaluate(self, ratings_df):
        """
        Evaluate the model using cross-validation.
        
        Returns:
            Dict with RMSE and MAE scores
        """
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings_df[["userId", "movieId", "rating"]],
            reader
        )

        # 5-fold cross-validation
        results = cross_validate(
            SVD(n_factors=100, n_epochs=20, random_state=42),
            data,
            measures=["RMSE", "MAE"],
            cv=5,
            verbose=True
        )

        metrics = {
            "rmse_mean": round(np.mean(results["test_rmse"]), 4),
            "rmse_std": round(np.std(results["test_rmse"]), 4),
            "mae_mean": round(np.mean(results["test_mae"]), 4),
            "mae_std": round(np.std(results["test_mae"]), 4),
        }

        print(f"\nCross-validation results:")
        print(f"  RMSE: {metrics['rmse_mean']} ± {metrics['rmse_std']}")
        print(f"  MAE:  {metrics['mae_mean']} ± {metrics['mae_std']}")

        return metrics

    def predict_rating(self, user_id, movie_id):
        """Predict a single user-movie rating."""
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est

    def recommend(self, user_id, n=10):
        """
        Get top-N movie recommendations for a user.
        
        Predicts ratings for all unseen movies and returns the highest predicted.
        
        Args:
            user_id: The user ID
            n: Number of recommendations
        
        Returns:
            DataFrame with movieId, title, genres, predicted_rating
        """
        # Get all movies the user has already rated
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        rated_movie_ids = set(user_ratings["movieId"].values)

        if len(user_ratings) == 0:
            print(f"User {user_id} not found. Returning popular movies instead.")
            return self._get_popular_movies(n)

        # Get all movie IDs
        all_movie_ids = set(self.movies_df["movieId"].values)

        # Movies the user hasn't rated
        unseen_movie_ids = all_movie_ids - rated_movie_ids

        # Predict ratings for all unseen movies
        predictions = []
        for movie_id in unseen_movie_ids:
            pred_rating = self.predict_rating(user_id, movie_id)
            predictions.append((movie_id, pred_rating))

        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:n]

        # Build result DataFrame
        top_movie_ids = [p[0] for p in top_predictions]
        top_scores = [round(p[1], 2) for p in top_predictions]

        result = self.movies_df[self.movies_df["movieId"].isin(top_movie_ids)].copy()
        score_map = dict(zip(top_movie_ids, top_scores))
        result["predicted_rating"] = result["movieId"].map(score_map)
        result = result.sort_values("predicted_rating", ascending=False)
        result = result[["movieId", "title", "genres", "predicted_rating"]].reset_index(drop=True)

        return result

    def _get_popular_movies(self, n=10):
        """Fallback: Return most popular movies by rating count and average."""
        movie_stats = self.ratings_df.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            num_ratings=("rating", "count")
        ).reset_index()

        # Filter movies with at least 50 ratings
        popular = movie_stats[movie_stats["num_ratings"] >= 50]
        popular = popular.sort_values("avg_rating", ascending=False).head(n)

        result = popular.merge(self.movies_df[["movieId", "title", "genres"]], on="movieId")
        result = result[["movieId", "title", "genres", "avg_rating"]].reset_index(drop=True)
        result.rename(columns={"avg_rating": "predicted_rating"}, inplace=True)

        return result


# ============================================================
# Standalone usage
# ============================================================
if __name__ == "__main__":
    from utils import load_movies, load_ratings

    # Load data
    movies = load_movies("small")
    ratings = load_ratings("small")

    # Build model
    recommender = CollaborativeRecommender(n_factors=50, n_epochs=20)
    recommender.fit(ratings, movies)

    # Evaluate
    print("\n--- Model Evaluation ---")
    metrics = recommender.evaluate(ratings)

    # Test: Recommendations for user 1
    print("\n--- Recommendations for User 1 ---")
    recs = recommender.recommend(user_id=1, n=10)
    print(recs.to_string(index=False))
