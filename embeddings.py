"""
Embedding model and FAISS index management.

This module loads the sentence embedding model, generates vector
representations for all documents, caches the embeddings to disk,
and creates a FAISS index for efficient similarity search.

The EmbeddingIndex class also provides methods for embedding user
queries before retrieval.
"""

import os
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, EMBEDDINGS_FILE
from data import documents


class EmbeddingIndex:

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.doc_texts = [doc["text"] for doc in documents]

        if os.path.exists(EMBEDDINGS_FILE):
            self.embeddings = np.load(EMBEDDINGS_FILE)
        else:
            self.embeddings = self.model.encode(
                self.doc_texts, convert_to_numpy=True
            ).astype("float32")

            np.save(EMBEDDINGS_FILE, self.embeddings)

        faiss.normalize_L2(self.embeddings)

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def embed_query(self, query):

        embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")

        faiss.normalize_L2(embedding)

        return embedding
