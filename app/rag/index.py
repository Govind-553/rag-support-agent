import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

from app.config import INDEX_DIR
from app.rag.loader import DocumentChunk
from app.rag.embeddings import EmbeddingModel

class FAISSIndex:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.embedding_model = None

    def _get_model(self) -> EmbeddingModel:
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel()
        return self.embedding_model

    def build(self, chunks: List[DocumentChunk]):
        """Builds FAISS index and stores chunk metadata."""
        if not chunks:
            print("No chunks provided to build FAISS index.")
            return

        model = self._get_model()
        texts = [f"{c.heading}\n{c.content}" for c in chunks]
        
        # Generate embeddings
        embeddings = model.embed_texts(texts)
        embeddings = np.array(embeddings).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        
        # Using IndexFlatIP for cosine similarity (inner product of normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.chunks = chunks

        # Save index and metadata
        self.save()

    def save(self):
        """Saves index and metadata chunks to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))

        # Save metadata chunks
        chunks_data = [c.to_dict() for c in self.chunks]
        with open(self.index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """Loads index and metadata chunks from disk."""
        index_file = self.index_dir / "index.faiss"
        chunks_file = self.index_dir / "chunks.json"

        if not index_file.exists() or not chunks_file.exists():
            return False

        # Load FAISS index
        self.index = faiss.read_index(str(index_file))

        # Load chunks
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            self.chunks = [
                DocumentChunk(
                    doc_id=c["doc_id"],
                    filename=c["filename"],
                    heading=c["heading"],
                    content=c["content"],
                    metadata=c["metadata"]
                )
                for c in chunks_data
            ]
        return True

    def search(self, query: str, k: int = 8) -> List[Tuple[DocumentChunk, float]]:
        """Searches the index and returns the top k matches with similarity scores."""
        if self.index is None:
            if not self.load():
                raise RuntimeError("FAISS index has not been built or loaded.")

        model = self._get_model()
        query_vector = model.embed_query(query).astype('float32')
        # Normalize query vector for cosine similarity
        query_vector = np.expand_dims(query_vector, axis=0)
        faiss.normalize_L2(query_vector)

        # Search index
        scores, indices = self.index.search(query_vector, min(k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
            
        return results
