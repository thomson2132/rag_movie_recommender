import chromadb
from chromadb.config import Settings
import numpy as np
from src.utils import setup_logging

logger = setup_logging()


class MovieRetriever:
    def __init__(self, config):
        self.config = config
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=config['chroma']['persist_directory'],
            anonymized_telemetry=False
        ))
        self.collection_name = config['chroma']['collection_name']
        self.top_k = config['rag']['top_k']

    def create_collection(self, embeddings, movie_data):
        """Create ChromaDB collection with movie embeddings in batches."""
        logger.info("Creating ChromaDB collection...")

        # Delete existing collection if exists
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except:
            pass

        # Create new collection
        collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Prepare data
        ids = [str(i) for i in range(len(embeddings))]
        documents = movie_data['combined_text'].tolist()
        metadatas = movie_data[['movie_title', 'genres', 'source']].to_dict('records')

        # Add embeddings in batches to avoid max batch size error
        batch_size = 5000  # Safely below ChromaDB's limit of 5461
        total_items = len(embeddings)

        logger.info(f"Adding {total_items} movies in batches of {batch_size}...")

        for i in range(0, total_items, batch_size):
            end_idx = min(i + batch_size, total_items)

            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_documents = documents[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]

            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

            logger.info(f"Added batch {i // batch_size + 1}: {end_idx}/{total_items} movies")

        logger.info(f"✓ Successfully added all {total_items} movies to ChromaDB")
        return collection

    def retrieve_similar_movies(self, query_embedding, n_results=None):
        """Retrieve similar movies based on query embedding."""
        if n_results is None:
            n_results = self.top_k

        collection = self.chroma_client.get_collection(self.collection_name)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        return results

    def search_by_text(self, query_text, embedding_model, n_results=None):
        """Search movies by text query."""
        # Generate query embedding
        query_embedding = embedding_model.encode(query_text, convert_to_tensor=True)

        # Retrieve similar movies
        results = self.retrieve_similar_movies(query_embedding.cpu().numpy(), n_results)

        return results
