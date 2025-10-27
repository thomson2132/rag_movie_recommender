import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
from tqdm import tqdm
from src.utils import setup_environment, setup_logging

logger = setup_logging()


class EmbeddingGenerator:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Use a better model for semantic search
        model_name = config['models'].get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')

        # Alternative better models for movie recommendations:
        # - "sentence-transformers/all-mpnet-base-v2" (better quality, slower)
        # - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (multilingual support)

        self.model = SentenceTransformer(model_name, device=str(device))
        logger.info(f"Loaded embedding model '{model_name}' on {device}")

    def generate_embeddings(self, texts, batch_size=32):
        """Generate embeddings for movie texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts...")

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for better cosine similarity
            )
            embeddings.append(batch_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def save_embeddings(self, embeddings, movie_data, save_path):
        """Save embeddings and metadata."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        data = {
            'embeddings': embeddings,
            'movie_data': movie_data
        }

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved embeddings to {save_path}")

    def load_embeddings(self, load_path):
        """Load embeddings and metadata."""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        logger.info(f"Loaded embeddings from {load_path}")
        return data['embeddings'], data['movie_data']
