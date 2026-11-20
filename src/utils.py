"""
Utility functions for the OTT Recommendation System.
Handles data loading, cleaning, and common transformations.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# Data Paths (using pathlib.Path for clean type-checking)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def load_movies(dataset_size: str = "small") -> pd.DataFrame:
    """
    Load movies dataset.

    Args:
        dataset_size: "small" for MovieLens 100K, "large" for MovieLens 25M

    Returns:
        DataFrame with columns: movieId, title, genres
    """
    if dataset_size == "small":
        path = RAW_DATA_DIR / "ml-latest-small" / "movies.csv"
    else:
        path = RAW_DATA_DIR / "ml-25m" / "movies.csv"

    if not path.exists():
        path = RAW_DATA_DIR / "movies.csv"

    movies = pd.read_csv(str(path))
    print(f"Loaded {len(movies)} movies from {path}")
    return movies


def load_ratings(dataset_size: str = "small") -> pd.DataFrame:
    """
    Load ratings dataset.

    Args:
        dataset_size: "small" for MovieLens 100K, "large" for MovieLens 25M

    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp
    """
    if dataset_size == "small":
        path = RAW_DATA_DIR / "ml-latest-small" / "ratings.csv"
    else:
        path = RAW_DATA_DIR / "ml-25m" / "ratings.csv"

    if not path.exists():
        path = RAW_DATA_DIR / "ratings.csv"

    ratings = pd.read_csv(str(path))
    print(f"Loaded {len(ratings)} ratings from {path}")
    return ratings


def load_tags(dataset_size: str = "small") -> pd.DataFrame:
    """
    Load tags dataset.

    Args:
        dataset_size: "small" for MovieLens 100K, "large" for MovieLens 25M

    Returns:
        DataFrame with columns: userId, movieId, tag, timestamp
    """
    if dataset_size == "small":
        path = RAW_DATA_DIR / "ml-latest-small" / "tags.csv"
    else:
        path = RAW_DATA_DIR / "ml-25m" / "tags.csv"

    if not path.exists():
        path = RAW_DATA_DIR / "tags.csv"

    if path.exists():
        tags = pd.read_csv(str(path))
        print(f"Loaded {len(tags)} tags from {path}")
        return tags
    else:
        print("Tags file not found, returning empty DataFrame")
        return pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])


def clean_data(movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
    """
    Clean and preprocess the data.

    - Convert timestamp to datetime
    - Extract year from movie title
    - Split genres into lists
    - Merge ratings with movies

    Returns:
        Tuple of (cleaned_movies, cleaned_ratings, merged_df)
    """
    ratings_clean = ratings_df.copy()
    ratings_clean["datetime"] = pd.to_datetime(ratings_clean["timestamp"], unit="s")
    ratings_clean["year_rated"] = ratings_clean["datetime"].dt.year

    movies_clean = movies_df.copy()
    movies_clean["release_year"] = movies_clean["title"].str.extract(r"\((\d{4})\)").astype(float)
    movies_clean["clean_title"] = movies_clean["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    movies_clean["genre_list"] = movies_clean["genres"].str.split("|")

    merged_df = ratings_clean.merge(movies_clean, on="movieId", how="inner")

    print(f"After cleaning: {len(movies_clean)} movies, {len(ratings_clean)} ratings, {len(merged_df)} merged rows")
    return movies_clean, ratings_clean, merged_df


def get_movie_stats(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-movie statistics.

    Returns:
        DataFrame with movieId, title, avg_rating, num_ratings, genres
    """
    stats = merged_df.groupby(["movieId", "title", "genres"]).agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count"),
        std_rating=("rating", "std")
    ).reset_index()

    stats["avg_rating"] = stats["avg_rating"].round(2)
    stats["std_rating"] = stats["std_rating"].fillna(0).round(2)
    stats = stats.sort_values("num_ratings", ascending=False)

    return stats


def get_user_stats(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-user statistics.

    Returns:
        DataFrame with userId, num_ratings, avg_rating, favorite_genre
    """
    user_stats = merged_df.groupby("userId").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()

    genre_exploded = merged_df.explode("genre_list")
    fav_genre = genre_exploded.groupby("userId")["genre_list"].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else "Unknown"
    ).reset_index()
    fav_genre.columns = ["userId", "favorite_genre"]

    user_stats = user_stats.merge(fav_genre, on="userId")
    user_stats["avg_rating"] = user_stats["avg_rating"].round(2)

    return user_stats


def save_processed_data(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, merged_df: pd.DataFrame):
    """Save cleaned data to processed directory."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    movies_df.to_csv(str(PROCESSED_DATA_DIR / "movies_cleaned.csv"), index=False)
    ratings_df.to_csv(str(PROCESSED_DATA_DIR / "ratings_cleaned.csv"), index=False)
    merged_df.to_csv(str(PROCESSED_DATA_DIR / "merged_data.csv"), index=False)

    print(f"Processed data saved to {PROCESSED_DATA_DIR}")
