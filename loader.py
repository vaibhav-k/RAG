"""
Document loader module for the RAG system.

Supports multiple document formats through a reader registry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader

from config import DOCS_PATH

logger = logging.getLogger(__name__)

# Type alias for document readers
Reader = Callable[[Path], str]


def read_txt(file_path: Path) -> str:
    """Read a UTF-8 text file."""
    return file_path.read_text(encoding="utf-8").strip()


def read_docx(file_path: Path) -> str:
    """Read text from a Word document."""

    document = Document(file_path)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


def read_markdown(file_path: Path) -> str:
    """Read text from a Markdown document."""
    return file_path.read_text(encoding="utf-8").strip()


def read_pdf(file_path: Path) -> str:
    """Read text from a PDF document."""
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_html(file_path: Path) -> str:
    """Read text from an HTML document."""
    soup = BeautifulSoup(
        file_path.read_text(encoding="utf-8"),
        "html.parser",
    )
    return soup.get_text(separator="\n")


# ------------------------------------------------------------------
# Register supported document readers here
# ------------------------------------------------------------------

READERS: dict[str, Reader] = {
    ".txt": read_txt,
    ".docx": read_docx,
    ".md": read_markdown,
    ".pdf": read_pdf,
    ".html": read_html,
}


def load_documents() -> list[dict[str, Any]]:
    """
    Load all supported documents from the docs directory.

    Raises:
        FileNotFoundError: If the document directory (DOCS_PATH) not found
        RuntimeError: If it is unable to read a file
    """
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"Document directory not found: {DOCS_PATH}")

    documents: list[dict[str, Any]] = []

    # Collect every supported file
    files: list[Path] = sorted(
        file
        for file in DOCS_PATH.iterdir()
        if file.is_file() and file.suffix.lower() in READERS
    )

    if not files:
        logger.warning("No supported documents found.")

    logger.info("Found %d document(s).", len(files))

    for file_path in files:
        reader = READERS.get(file_path.suffix.lower())

        if reader is None:
            logger.debug(f"Skipping unsupported file: {file_path.name}")
            continue

        try:
            text = reader(file_path)
        except Exception as exc:
            logger.exception(f"Failed reading {file_path.name}")
            raise RuntimeError(f"Unable to read {file_path.name}") from exc

        if not text.strip():
            logger.warning(f"Skipping empty document: {file_path.name}")
            continue

        documents.append(
            {
                "id": file_path.name,
                "title": file_path.stem.replace("_", " ").title(),
                "text": text,
                "metadata": {
                    "source": file_path.name,
                    "file_type": file_path.suffix.lower(),
                },
            }
        )

    logger.info(f"Loaded {len(documents)} document(s).")

    return documents
