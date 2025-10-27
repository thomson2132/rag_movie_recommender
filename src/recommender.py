import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import DataLoader
from src.embeddings import EmbeddingGenerator
from src.retriever import MovieRetriever
from src.groq_generator import GroqGenerator
from src.utils import load_config, setup_environment, setup_logging

logger = setup_logging()


class RAGMovieRecommender:
    def __init__(self):
        self.config = load_config()
        self.device = setup_environment()

        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config, self.device)
        self.retriever = MovieRetriever(self.config)
        self.groq_generator = GroqGenerator(self.config)

        self.movie_data = None
        self.ratings_data = None

    def setup_system(self):
        """Setup the complete RAG system."""
        logger.info("Setting up RAG Movie Recommender System...")

        # Load and combine datasets
        self.movie_data, self.ratings_data = self.data_loader.combine_datasets()

        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(
            self.movie_data['combined_text'].tolist()
        )

        # Save embeddings
        embeddings_path = f"{self.config['data']['processed_path']}/embeddings.pkl"
        self.embedding_generator.save_embeddings(
            embeddings, self.movie_data, embeddings_path
        )

        # Create ChromaDB collection
        self.retriever.create_collection(embeddings, self.movie_data)

        logger.info("System setup complete!")

    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        """Get collaborative filtering recommendations."""
        if self.ratings_data is None:
            return []

        # Simple collaborative filtering using user-item matrix
        user_movie_matrix = self.ratings_data.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)

        if user_id not in user_movie_matrix.index:
            return []

        # Calculate user similarity
        user_similarity = cosine_similarity(user_movie_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_movie_matrix.index,
            columns=user_movie_matrix.index
        )

        # Get similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]

        # Get recommendations from similar users
        recommendations = []
        for similar_user in similar_users.index:
            user_ratings = self.ratings_data[self.ratings_data['userId'] == similar_user]
            top_rated = user_ratings.nlargest(n_recommendations, 'rating')
            recommendations.extend(top_rated['movieId'].tolist())

        return list(set(recommendations))[:n_recommendations]

    def recommend(self, user_query, user_id=None, use_collaborative=True):
        """Generate hybrid recommendations."""
        logger.info(f"Processing recommendation request: {user_query}")

        # RAG-based retrieval
        rag_results = self.retriever.search_by_text(
            user_query,
            self.embedding_generator.model
        )

        # Extract retrieved movies
        retrieved_movies = []
        for i, metadata in enumerate(rag_results['metadatas'][0]):
            retrieved_movies.append(metadata)

        # Generate recommendations using Groq
        recommendations = self.groq_generator.generate_recommendations(
            user_query, retrieved_movies
        )

        # Add collaborative filtering if user_id provided
        if use_collaborative and user_id:
            collab_recs = self.get_collaborative_recommendations(user_id)
            collab_info = f"\n\nAdditional recommendations based on similar users:\n"
            collab_info += ", ".join([str(mid) for mid in collab_recs[:5]])
            recommendations += collab_info

        return recommendations, retrieved_movies
