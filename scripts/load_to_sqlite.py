import argparse
import sqlite3
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_DB = ROOT_DIR / "data" / "bitcoin_project.db"


def load_csv(path, chunksize=100_000):
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV input: {path}")
    return pd.read_csv(path, chunksize=chunksize, low_memory=False)


def create_tables(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tweets (
            tweet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            user_location TEXT,
            user_description TEXT,
            user_created TEXT,
            user_followers REAL,
            user_friends REAL,
            user_favourites REAL,
            user_verified INTEGER,
            date TEXT,
            text TEXT,
            hashtags TEXT,
            source TEXT,
            is_retweet INTEGER,
            sentiment_positive REAL,
            sentiment_negative REAL,
            sentiment_neutral REAL,
            sentiment_compound REAL,
            sentiment_label TEXT,
            UNIQUE(date, text)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hourly_sentiment (
            hour TEXT PRIMARY KEY,
            tweet_count REAL,
            average_sentiment REAL,
            median_sentiment REAL,
            positive_share REAL,
            negative_share REAL,
            average_followers REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hourly_prices (
            hour TEXT PRIMARY KEY,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            hour TEXT,
            horizon_hours INTEGER,
            model TEXT,
            fold INTEGER,
            actual_return REAL,
            predicted_return REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            horizon_hours INTEGER,
            model TEXT,
            MAE REAL,
            RMSE REAL,
            directional_accuracy REAL,
            observations INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS x_posts_live (
            tweet_id TEXT,
            created_at TEXT,
            text TEXT,
            like_count REAL,
            repost_count REAL,
            reply_count REAL,
            view_count REAL,
            author_followers REAL,
            author_username TEXT,
            post_url TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_articles (
            provider TEXT,
            title TEXT,
            url TEXT,
            published_date TEXT,
            source TEXT,
            content TEXT,
            retrieved_at TEXT
        )
        """
    )


def import_csv(conn, table_name, csv_path, columns=None):
    first_chunk = True
    for df in load_csv(csv_path):
        if columns is not None:
            df = df[columns]
        df.to_sql(
            table_name,
            conn,
            if_exists="replace" if first_chunk else "append",
            index=False,
        )
        first_chunk = False
    if first_chunk:
        raise ValueError(f"No records to import from {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Load the processed project CSV files into SQLite.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite database")
    parser.add_argument("--reset", action="store_true", help="Rebuild the schema from scratch")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace imported tables in an existing database",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if db_path.exists() and db_path.stat().st_size and not (
        args.replace_existing or args.reset
    ):
        raise RuntimeError(
            f"Database already exists and will not be overwritten: {db_path.resolve()}. "
            "Pass --replace-existing only if replacing imported tables is intended."
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        if args.reset:
            for table in [
                "tweets", "hourly_sentiment", "hourly_prices",
                "model_predictions", "evaluation_runs", "x_posts_live",
                "news_articles",
            ]:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
        create_tables(conn)

        tweet_file = DATA_DIR / "bitcoin_tweets_with_sentiment.csv"
        if tweet_file.exists():
            import_csv(conn, "tweets", tweet_file)
        else:
            fallback = DATA_DIR / "bitcoin_tweets_clean_sample.csv"
            if fallback.exists():
                import_csv(conn, "tweets", fallback)
            else:
                raise FileNotFoundError("No tweet CSV found for SQLite import.")

        import_csv(conn, "hourly_sentiment", DATA_DIR / "hourly_sentiment.csv")
        import_csv(conn, "hourly_prices", DATA_DIR / "bitcoin_historical_hourly.csv")

        predictions_path = ROOT_DIR / "outputs" / "walk_forward_predictions.csv"
        if predictions_path.exists():
            import_csv(conn, "model_predictions", predictions_path)

        metrics_path = ROOT_DIR / "outputs" / "walk_forward_metrics.csv"
        if metrics_path.exists():
            import_csv(conn, "evaluation_runs", metrics_path)

        apify_file = DATA_DIR / "x_posts_apify.csv"
        if apify_file.exists():
            import_csv(conn, "x_posts_live", apify_file)

        news_dir = ROOT_DIR / "data" / "raw" / "news"
        for provider in ("tavily", "exa"):
            news_file = news_dir / f"{provider}_news.csv"
            if news_file.exists():
                news = pd.read_csv(news_file)
                news.insert(0, "provider", provider)
                news.to_sql(
                    "news_articles",
                    conn,
                    if_exists="replace" if provider == "tavily" else "append",
                    index=False,
                )
    print(f"SQLite database written to: {db_path.resolve()}")


if __name__ == "__main__":
    main()
