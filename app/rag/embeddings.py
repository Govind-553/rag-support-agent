from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from app.config import EMBEDDING_MODEL_NAME

class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        # Local model loading using sentence-transformers
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates embedding vectors for a list of texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """Generates embedding vector for a single query."""
        # For BGE models, they recommend query prefixing or standard encoding.
        # BGE small v1.5 works fine with standard encode, but we can prefix query if needed.
        # Let's keep it simple and standard:
        return self.model.encode(text, convert_to_numpy=True)
