"""
Embedding and vector index module for RAG.

This module:
- Loads documents from disk
- Splits them into chunks
- Converts chunks into embeddings using SentenceTransformers
- Stores embeddings in a FAISS index for similarity search
"""

import os
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from loader import load_documents
from text_splitter import chunk_text
from config import EMBEDDING_MODEL_NAME, EMBEDDINGS_FILE


class EmbeddingIndex:
    """
    Builds and manages document embeddings and FAISS index.
    """

    def __init__(self):
        """
        Initialize embedding model and build FAISS index.
        """

        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        raw_documents = load_documents()

        self.documents = []

        # Build chunks
        for doc in raw_documents:
            chunks = chunk_text(doc["text"])

            for i, chunk in enumerate(chunks):
                self.documents.append(
                    {
                        "id": f"{doc['id']}_chunk_{i}",
                        "title": doc["title"],
                        "text": chunk,
                        "metadata": doc.get("metadata", {}),
                    }
                )

        # Load or create embeddings
        if os.path.exists(EMBEDDINGS_FILE):
            self.embeddings = np.load(EMBEDDINGS_FILE)

        else:
            texts = [doc["text"] for doc in self.documents]

            self.embeddings = self.model.encode(texts, convert_to_numpy=True).astype(
                "float32"
            )

            np.save(EMBEDDINGS_FILE, self.embeddings)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert a query into an embedding vector.

        Args:
            query: User question

        Returns:
            Normalized query embedding
        """

        embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")

        faiss.normalize_L2(embedding)

        return embedding
