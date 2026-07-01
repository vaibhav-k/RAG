"""
Application configuration.

This module centralizes all configurable constants used throughout the
RAG application, including models, retrieval settings, generation
parameters, embedding cache, FAISS index persistence, and text
chunking configuration.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DOCS_PATH = BASE_DIR / "docs"

CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

EMBEDDINGS_FILE = CACHE_DIR / "doc_embeddings.npy"
FAISS_INDEX_FILE = CACHE_DIR / "faiss.index"
CACHE_METADATA_FILE = CACHE_DIR / "cache_metadata.json"

# =============================================================================
# Models
# =============================================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-small"

# =============================================================================
# Embedding Configuration
# =============================================================================

EMBEDDING_BATCH_SIZE = 64

# Automatically use GPU if available
DEVICE = "cuda"

# =============================================================================
# Chunking
# =============================================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# =============================================================================
# Retrieval
# =============================================================================

TOP_K = 5

SIMILARITY_THRESHOLD = 0.0

MAX_RETRIEVED_DOCS = 5

MAX_CONTEXT_CHARS = 6000

# =============================================================================
# Generation
# =============================================================================

TOKENIZER_MAX_LENGTH = 512

MAX_NEW_TOKENS = 50

DO_SAMPLE = False

# =============================================================================
# Prompt
# =============================================================================

PROMPT_TEMPLATE = """
You are an Amazon customer support assistant.

Use the context to give a clear, complete answer.

If the context is partial, combine all relevant details.

Do not be overly short.

Context:
{context}

Question:
{question}

Answer:
"""
