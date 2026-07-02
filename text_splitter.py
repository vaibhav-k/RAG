"""
Text splitting utilities for the RAG system.

This module splits long documents into overlapping chunks suitable for
embedding and semantic retrieval.
"""

from __future__ import annotations

import logging
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)
_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])

# Add sentence boundary detector
if "sentencizer" not in _nlp.pipe_names:
    _nlp.add_pipe("sentencizer")


def chunk_text_hierarchical(
    text: str,
    max_chunk_chars: int = 1200,
    distance_threshold: float = 0.75,
) -> list[str]:
    """
    Split text into semantic chunks using hierarchical clustering.

    Args:
        text: Input document.
        max_chunk_chars: Maximum chunk length.
        distance_threshold: Smaller = more chunks.
    """
    text = text.strip()

    if not text:
        return []

    doc = _nlp(text)
    sentences = [sent.text for sent in doc.sents]

    if len(sentences) <= 1:
        return [text]

    embeddings = _MODEL.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    clustering = AgglomerativeClustering(
        None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )

    labels = clustering.fit_predict(embeddings)

    semantic_chunks = []

    current = [sentences[0]]
    current_label = labels[0]

    # Merge consecutive sentences belonging to the same semantic cluster
    for sentence, label in zip(sentences[1:], labels[1:]):
        if label == current_label:
            current.append(sentence)
        else:
            semantic_chunks.append(" ".join(current))
            current = [sentence]
            current_label = label

    semantic_chunks.append(" ".join(current))

    # Ensure chunks are not too large
    final_chunks = []

    for chunk in semantic_chunks:

        if len(chunk) <= max_chunk_chars:
            final_chunks.append(chunk)
            continue

        # Fall back to character-based splitting for exceptionally large clusters
        start = 0
        while start < len(chunk):
            final_chunks.append(chunk[start : start + max_chunk_chars])
            start += max_chunk_chars

    logger.info(f"Split document into {len(final_chunks)} semantic chunks.")

    return final_chunks


def chunk_text_fixed_chunk_size(
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
