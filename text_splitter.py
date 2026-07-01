"""
Text splitting utilities for RAG.

This module handles splitting long documents into smaller overlapping
chunks so that embeddings and retrieval work more effectively.
"""

from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split a text into overlapping chunks.

    Args:
        text: Input document text
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks
