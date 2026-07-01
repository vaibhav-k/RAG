"""
Application configuration.

This module contains all configurable constants used throughout the RAG
application, including model names, retrieval settings, generation
parameters, embedding cache location, and the prompt template.

Keeping configuration in one place makes the application easier to
maintain and experiment with.
"""

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Generator model
GENERATOR_MODEL_NAME = "google/flan-t5-small"

# Retrieval settings
TOP_K = 1
SIMILARITY_THRESHOLD = 0.25

# Generation settings
MAX_NEW_TOKENS = 50
DO_SAMPLE = False

# Embedding cache
EMBEDDINGS_FILE = "doc_embeddings.npy"

# Prompt template
PROMPT_TEMPLATE = """
You are an Amazon customer support assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

# Maximum number of retrieved documents
MAX_RETRIEVED_DOCS = 5

# Maximum context length (characters)
MAX_CONTEXT_CHARS = 6000

# Source for the documentation txt files
DOCS_PATH = "docs"
