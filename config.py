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
TOP_K = 3
SIMILARITY_THRESHOLD = 0.45

# Generation settings
MAX_NEW_TOKENS = 50
DO_SAMPLE = False

# Embedding cache
EMBEDDINGS_FILE = "doc_embeddings.npy"

# Prompt template
PROMPT_TEMPLATE = """
You are a helpful Amazon customer support assistant.

Use ONLY the information provided in the context below to answer the question.

If the answer is not contained in the context, reply with:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""
