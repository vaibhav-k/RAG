"""
Text splitting utilities for the RAG system.

This module splits long documents into overlapping chunks suitable for
embedding and semantic retrieval.
"""

from __future__ import annotations

import logging

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text:
            Input text.

        chunk_size:
            Maximum number of characters per chunk.

        overlap:
            Number of overlapping characters.

    Returns:
        List of text chunks.

    Raises:
        ValueError:
            If chunk_size or overlap are invalid.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = text.strip()

    if not text:
        logger.warning("Received empty text for chunking.")
        return []

    chunks: list[str] = []

    step = chunk_size - overlap
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    logger.debug(
        "Split text (%d chars) into %d chunks.",
        len(text),
        len(chunks),
    )

    return chunks
