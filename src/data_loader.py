import pandas as pd
import os
import json
from pathlib import Path
from src.utils import load_config, setup_logging

logger = setup_logging()


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.ml25m_path = config['data']['ml25m_path']
        self.netflix_path = config['data']['netflix_path']
        self.tmdb_path = config['data']['tmdb_path']

    def parse_json_genres(self, genre_string):
        """Parse TMDB JSON genre format to readable string."""
        try:
            if pd.isna(genre_string) or genre_string == '':
                return ''

            # Parse JSON string
            genres = json.loads(genre_string)

            # Extract genre names
            if isinstance(genres, list):
                genre_names = [g.get('name', '') for g in genres if isinstance(g, dict)]
                return '|'.join(genre_names)

            return ''
        except:
            return str(genre_string) if not pd.isna(genre_string) else ''

    def load_ml25m_data(self):
        """Load MovieLens 25M dataset."""
        logger.info("Loading MovieLens 25M data...")
        try:
            movies = pd.read_csv(os.path.join(self.ml25m_path, 'movies.csv'), encoding='utf-8')
            ratings = pd.read_csv(os.path.join(self.ml25m_path, 'ratings.csv'), encoding='utf-8')

            try:
                tags = pd.read_csv(os.path.join(self.ml25m_path, 'tags.csv'), encoding='utf-8')
            except:
                tags = pd.DataFrame()

            logger.info(f"Loaded {len(movies)} movies and {len(ratings)} ratings from ML-25M")
            return movies, ratings, tags
        except Exception as e:
            logger.error(f"Error loading ML-25M data: {e}")
            return None, None, None

    def load_netflix_data(self):
        """Load Netflix dataset."""
        logger.info("Loading Netflix data...")
        try:
            netflix_titles = pd.read_csv(
                os.path.join(self.netflix_path, 'netflix_titles.csv'),
                encoding='utf-8'
            )
            logger.info(f"Loaded {len(netflix_titles)} Netflix titles")
            return netflix_titles
        except Exception as e:
            logger.error(f"Error loading Netflix data: {e}")
            return None

    def load_tmdb_data(self):
        """Load TMDB dataset."""
        logger.info("Loading TMDB data...")
        try:
            tmdb_movies = pd.read_csv(
                os.path.join(self.tmdb_path, 'tmdb_5000_movies.csv'),
                encoding='utf-8'
            )

            try:
                tmdb_credits = pd.read_csv(
                    os.path.join(self.tmdb_path, 'tmdb_5000_credits.csv'),
                    encoding='utf-8'
                )
            except:
                tmdb_credits = pd.DataFrame()

            logger.info(f"Loaded {len(tmdb_movies)} TMDB movies")
            return tmdb_movies, tmdb_credits
        except Exception as e:
            logger.error(f"Error loading TMDB data: {e}")
            return None, None

    def combine_datasets(self):
        """Combine and clean all datasets."""
        logger.info("Combining datasets...")

        # Load all datasets
        ml_movies, ml_ratings, ml_tags = self.load_ml25m_data()
        netflix_titles = self.load_netflix_data()
        tmdb_movies, tmdb_credits = self.load_tmdb_data()

        combined_dfs = []

        # Process MovieLens data
        if ml_movies is not None:
            logger.info("Processing MovieLens data...")
            ml_df = ml_movies.copy()
            ml_df['source'] = 'movielens'
            ml_df = ml_df.rename(columns={'movieId': 'id', 'title': 'movie_title'})

            # MovieLens genres are already pipe-separated (Action|Adventure)
            if 'genres' not in ml_df.columns:
                ml_df['genres'] = ''

            ml_df['description'] = ''
            ml_df['genres'] = ml_df['genres'].fillna('')

            combined_dfs.append(ml_df[['movie_title', 'genres', 'description', 'source']])

        # Process Netflix data
        if netflix_titles is not None:
            logger.info("Processing Netflix data...")
            netflix_df = netflix_titles.copy()

            # Select relevant columns
            cols_to_keep = []
            if 'title' in netflix_df.columns:
                cols_to_keep.append('title')
            if 'listed_in' in netflix_df.columns:
                cols_to_keep.append('listed_in')
            if 'description' in netflix_df.columns:
                cols_to_keep.append('description')

            if cols_to_keep:
                netflix_df = netflix_df[cols_to_keep].copy()
                netflix_df['source'] = 'netflix'

                # Rename columns
                if 'title' in netflix_df.columns:
                    netflix_df = netflix_df.rename(columns={'title': 'movie_title'})
                if 'listed_in' in netflix_df.columns:
                    netflix_df = netflix_df.rename(columns={'listed_in': 'genres'})

                # Ensure all columns exist
                if 'description' not in netflix_df.columns:
                    netflix_df['description'] = ''
                if 'genres' not in netflix_df.columns:
                    netflix_df['genres'] = ''

                netflix_df['description'] = netflix_df['description'].fillna('')
                netflix_df['genres'] = netflix_df['genres'].fillna('')

                # Netflix uses comma-separated genres, convert to pipe-separated
                netflix_df['genres'] = netflix_df['genres'].str.replace(',', '|')

                combined_dfs.append(netflix_df[['movie_title', 'genres', 'description', 'source']])

        # Process TMDB data
        if tmdb_movies is not None:
            logger.info("Processing TMDB data...")
            tmdb_df = tmdb_movies.copy()

            # Parse JSON genres column
            if 'genres' in tmdb_df.columns:
                logger.info("Parsing TMDB JSON genres...")
                tmdb_df['genres'] = tmdb_df['genres'].apply(self.parse_json_genres)
            else:
                tmdb_df['genres'] = ''

            tmdb_df['source'] = 'tmdb'

            # Rename columns
            if 'title' in tmdb_df.columns:
                tmdb_df = tmdb_df.rename(columns={'title': 'movie_title'})
            if 'overview' in tmdb_df.columns:
                tmdb_df = tmdb_df.rename(columns={'overview': 'description'})

            # Ensure all columns exist
            if 'description' not in tmdb_df.columns:
                tmdb_df['description'] = ''

            tmdb_df['description'] = tmdb_df['description'].fillna('')
            tmdb_df['genres'] = tmdb_df['genres'].fillna('')

            combined_dfs.append(tmdb_df[['movie_title', 'genres', 'description', 'source']])

        # Combine all dataframes
        if combined_dfs:
            combined_df = pd.concat(combined_dfs, ignore_index=True)

            # Clean combined dataset
            combined_df = self.clean_data(combined_df)

            logger.info(f"Combined dataset contains {len(combined_df)} movies")

            # Save processed data
            os.makedirs(self.config['data']['processed_path'], exist_ok=True)
            combined_df.to_csv(
                os.path.join(self.config['data']['processed_path'], 'combined_movies.csv'),
                index=False,
                encoding='utf-8'
            )

            return combined_df, ml_ratings
        else:
            logger.error("No datasets were successfully loaded")
            return None, None

    def clean_data(self, df):
        """Clean and preprocess the combined dataset."""
        logger.info("Cleaning combined dataset...")

        # Remove duplicates based on movie title
        original_count = len(df)
        df = df.drop_duplicates(subset=['movie_title'], keep='first')
        logger.info(f"Removed {original_count - len(df)} duplicate movies")

        # Handle missing values
        df['description'] = df['description'].fillna('')
        df['genres'] = df['genres'].fillna('')
        df['movie_title'] = df['movie_title'].fillna('Unknown')

        # Remove any remaining JSON-like strings in genres
        df['genres'] = df['genres'].apply(lambda x: x if not x.startswith('[{') else '')

        # Create combined text for embedding
        df['combined_text'] = (
                df['movie_title'].astype(str) + ' ' +
                df['genres'].astype(str).replace('|', ' ') + ' ' +
                df['description'].astype(str)
        )

        # Remove movies with empty titles
        df = df[df['movie_title'] != 'Unknown']
        df = df[df['movie_title'] != '']

        logger.info(f"Final cleaned dataset: {len(df)} movies")

        return df
