"""
Embedding model and FAISS index management.

This module:
- Loads the sentence embedding model.
- Splits documents into overlapping chunks.
- Generates embeddings for chunks.
- Caches embeddings.
- Builds a FAISS index.
"""

import os
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_FILE,
)

from data import documents


class EmbeddingIndex:
    """Manages document chunking, embeddings, and the FAISS index."""

    # Adjust these values based on your documents
    CHUNK_SIZE = 500  # characters
    CHUNK_OVERLAP = 100  # characters

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Build chunks
        self.documents = self._build_chunks(documents)

        self.doc_texts = [doc["text"] for doc in self.documents]

        # Load or create embeddings
        if os.path.exists(EMBEDDINGS_FILE):
            self.embeddings = np.load(EMBEDDINGS_FILE)
        else:
            self.embeddings = self.model.encode(
                self.doc_texts,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).astype("float32")

            np.save(EMBEDDINGS_FILE, self.embeddings)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def _build_chunks(self, documents):
        """
        Split documents into overlapping chunks while preserving metadata.
        """

        chunks = []

        for document in documents:

            text = document["text"]

            start = 0

            while start < len(text):

                end = start + self.CHUNK_SIZE

                chunk = document.copy()

                chunk["text"] = text[start:end]

                chunk["chunk_start"] = start
                chunk["chunk_end"] = min(end, len(text))

                chunks.append(chunk)

                start += self.CHUNK_SIZE - self.CHUNK_OVERLAP

        return chunks

    def embed_query(self, query):
        """
        Embed a user query.
        """

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
        ).astype("float32")

        faiss.normalize_L2(embedding)

        return embedding
