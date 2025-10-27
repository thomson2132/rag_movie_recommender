# create_sample_data.py
import pandas as pd
import os

# Load your full datasets
movies = pd.read_csv('data/raw/ml-25m/movies.csv')
netflix = pd.read_csv('data/raw/netflix/netflix_titles.csv')
tmdb = pd.read_csv('data/raw/tmdb/tmdb_5000_movies.csv')

# Take samples
movies_sample = movies.head(1000)  # 1000 movies instead of 62K
netflix_sample = netflix.head(500)
tmdb_sample = tmdb.head(500)

# Save samples
os.makedirs('data/sample/', exist_ok=True)
movies_sample.to_csv('data/sample/movies_sample.csv', index=False)
netflix_sample.to_csv('data/sample/netflix_sample.csv', index=False)
tmdb_sample.to_csv('data/sample/tmdb_sample.csv', index=False)

print("✓ Sample datasets created in data/sample/")
