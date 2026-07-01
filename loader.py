"""
Document loader module for the RAG system.

This module loads raw text documents from the local `docs/` directory
and converts them into a structured format that can be used by the
embedding and retrieval pipeline.

Each file is treated as a separate document.
"""

import os
from typing import List, Dict

from config import DOCS_PATH


def load_documents() -> List[Dict]:
    """
    Load all text documents from the docs directory.

    Each document is converted into a dictionary with:
        - id: filename
        - title: derived from filename
        - text: full file content

    Raises:
        FileNotFoundError: If the docs directory is absent.

    Returns:
        List of document dictionaries.
    """
    documents = []

    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"Docs directory not found: {DOCS_PATH}")

    for file_name in os.listdir(DOCS_PATH):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(DOCS_PATH, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        documents.append(
            {
                "id": file_name,
                "title": file_name.replace(".txt", "").title(),
                "text": text,
                "metadata": {"source": file_name},
            }
        )

    return documents
