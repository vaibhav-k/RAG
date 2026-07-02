"""
Embedding and vector index module for RAG.

This module:
- Loads documents
- Splits them into chunks
- Generates embeddings
- Stores/retrieves FAISS index with caching support
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDINGS_FILE,
    FAISS_INDEX_FILE,
    CACHE_METADATA_FILE,
    DEVICE,
)

from loader import load_documents
from text_splitter import chunk_text_fixed_chunk_size

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """
    Builds and manages document embeddings and FAISS index.
    """

    def __init__(self) -> None:
        """
        Initialize embedding model and prepare cache/index state.
        """

        # ----------------------------
        # Device selection
        # ----------------------------
        self.device: str = (
            "cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        )

        logger.info("Using device: %s", self.device)

        # ----------------------------
        # Load embedding model
        # ----------------------------
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=self.device,
        )

        # ----------------------------
        # Load documents
        # ----------------------------
        raw_documents = load_documents()

        self.documents: list[dict[str, Any]] = []

        for doc in raw_documents:
            chunks = chunk_text_fixed_chunk_size(doc["text"])

            for i, chunk in enumerate(chunks):
                self.documents.append(
                    {
                        "id": f"{doc['id']}_chunk_{i}",
                        "title": doc["title"],
                        "text": chunk,
                        "metadata": doc.get("metadata", {}),
                    }
                )

        logger.info("Created %d document chunks.", len(self.documents))

        # ----------------------------
        # Cache metadata (for validation)
        # ----------------------------
        self.cache_metadata: dict[str, Any] = {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "device": self.device,
            "num_chunks": len(self.documents),
        }

        self.embeddings: np.ndarray | None = None
        self.index: faiss.Index | None = None

    # ============================================================
    # Cache utilities
    # ============================================================

    def _cache_valid(self) -> bool:
        """
        Check whether cached embeddings and FAISS index are valid.
        """

        if not CACHE_METADATA_FILE.exists():
            return False

        try:
            content = CACHE_METADATA_FILE.read_text(encoding="utf-8").strip()

            if not content:
                return False

            cached = json.loads(content)

            return cached == self.cache_metadata
        except Exception as e:
            logger.warning(f"Cache validation failed: {e}")
            return False

    # ============================================================
    # Build / load index
    # ============================================================

    def _load_cache(self) -> bool:
        """
        Try loading FAISS index and embeddings from disk.
        Returns True if successful.
        """

        if not (EMBEDDINGS_FILE.exists() and FAISS_INDEX_FILE.exists()):
            return False

        try:
            logger.info("Loading embeddings from cache...")

            self.embeddings = np.load(EMBEDDINGS_FILE)

            self.index = faiss.read_index(str(FAISS_INDEX_FILE))

            return True

        except Exception as e:
            logger.warning("Failed to load cache: %s", e)
            return False

    def _save_cache(self) -> None:
        """
        Save embeddings, FAISS index, and metadata.
        """

        try:
            logger.info("Saving embeddings and FAISS index...")

            np.save(EMBEDDINGS_FILE, self.embeddings)

            faiss.write_index(
                self.index,
                str(FAISS_INDEX_FILE),
            )

            with open(CACHE_METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        **self.cache_metadata,
                        "created_at": datetime.utcnow().isoformat(),
                    },
                    f,
                    indent=2,
                )

        except Exception as e:
            logger.error("Failed to save cache: %s", e)

    # ============================================================
    # Embedding generation
    # ============================================================

    def _create_embeddings(self) -> np.ndarray:
        """
        Generate embeddings using batching.
        """

        texts = [doc["text"] for doc in self.documents]

        all_embeddings = []

        logger.info("Generating embeddings...")

        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]

            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")

            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        return embeddings

    # ============================================================
    # Build index
    # ============================================================

    def _build_index(self) -> None:
        """
        Build FAISS index from embeddings.
        """

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    # ============================================================
    # Public initialization hook
    # ============================================================

    def initialize(self) -> None:
        """
        Initialize embeddings + FAISS index (cache-aware).
        """

        try:
            # Try cache first
            if self._cache_valid() and self._load_cache():
                logger.info("Using cached FAISS index.")
                return

            logger.info("Cache invalid or missing. Rebuilding index...")

            self.embeddings = self._create_embeddings()

            self._build_index()

            self._save_cache()

            logger.info("Index built successfully.")

        except Exception as e:
            logger.exception("Failed to initialize embeddings index.")
            raise RuntimeError("EmbeddingIndex initialization failed") from e

    # ============================================================
    # Query embedding
    # ============================================================

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert query into normalized embedding.
        """

        try:
            embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
            ).astype("float32")

            faiss.normalize_L2(embedding)

            return embedding

        except Exception as e:
            logger.exception("Query embedding failed.")
            raise RuntimeError("Failed to embed query") from e
