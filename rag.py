"""
Retrieval-Augmented Generation (RAG) pipeline.

Coordinates document retrieval and language generation.
"""

from typing import Any, Dict, List, Optional, Tuple

from config import (
    PROMPT_TEMPLATE,
    MAX_CONTEXT_CHARS,
    MAX_RETRIEVED_DOCS,
)

from retriever import Retriever
from generator import Generator
from embeddings import EmbeddingIndex

# Type alias for clarity
RetrievedDoc = Dict[str, Any]


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self) -> None:
        """Initialize embedding index, retriever, and generator."""
        self.embedding_index: EmbeddingIndex = EmbeddingIndex()
        self.embedding_index.initialize()

        self.retriever: Retriever = Retriever(self.embedding_index)
        self.generator: Generator = Generator()

    def answer(
        self,
        question: str,
    ) -> Tuple[Optional[List[RetrievedDoc]], str]:
        """
        Generate an answer for a user question using RAG.

        Args:
            question: User query string.

        Returns:
            Tuple of:
                - Retrieved documents (or None if failed)
                - Generated answer string
        """

        try:
            retrieved_docs, _ = self.retriever.retrieve(question)
        except Exception as e:
            return None, f"Document retrieval failed: {e}"

        if not retrieved_docs:
            return None, "Sorry, I couldn't find any relevant documents."

        retrieved_docs = retrieved_docs[:MAX_RETRIEVED_DOCS]

        context_parts: List[str] = []
        current_length: int = 0

        for doc in retrieved_docs:
            document = doc["document"]
            text: str = document["text"].strip()

            if current_length + len(text) > MAX_CONTEXT_CHARS:
                remaining: int = MAX_CONTEXT_CHARS - current_length
                if remaining > 0:
                    context_parts.append(text[:remaining])
                break

            context_parts.append(f"[Source: {document['title']}]\n{text}")

            current_length += len(text)

        context: str = "\n\n".join(context_parts)

        prompt: str = PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        try:
            answer: str = self.generator.generate(prompt)
        except Exception as e:
            return retrieved_docs, f"Answer generation failed: {e}"

        return retrieved_docs, answer
