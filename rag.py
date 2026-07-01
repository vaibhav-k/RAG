"""
Retrieval-Augmented Generation (RAG) pipeline.

Coordinates document retrieval and language generation.
"""

from typing import List, Tuple, Optional

from config import (
    PROMPT_TEMPLATE,
    MAX_CONTEXT_CHARS,
    MAX_RETRIEVED_DOCS,
)

from retriever import Retriever
from generator import Generator
from embeddings import EmbeddingIndex


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self):
        self.embedding_index = EmbeddingIndex()
        self.retriever = Retriever(self.embedding_index)
        self.generator = Generator()

    def answer(self, question: str):

        try:
            retrieved_docs, _ = self.retriever.retrieve(question)
        except Exception as e:
            return None, f"Document retrieval failed: {e}"

        if not retrieved_docs:
            return None, "Sorry, I couldn't find any relevant documents."

        retrieved_docs = retrieved_docs[:MAX_RETRIEVED_DOCS]

        # ✅ Build context correctly (ONLY ONCE)
        context_parts = []
        current_length = 0

        for doc in retrieved_docs:
            text = doc["document"]["text"].strip()

            if current_length + len(text) > MAX_CONTEXT_CHARS:
                remaining = MAX_CONTEXT_CHARS - current_length
                if remaining > 0:
                    context_parts.append(text[:remaining])
                break

            context_parts.append(f"""Title: {doc["document"]["title"]}
Category: {doc["document"]["category"]}
Source: {doc["document"]["source"]}

{text}""")

            current_length += len(text)

        context = "\n\n".join(context_parts)

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        print("\n===== PROMPT =====\n")
        print(prompt)
        print("\n==================\n")

        try:
            answer = self.generator.generate(prompt)
        except Exception as e:
            return retrieved_docs, f"Answer generation failed: {e}"

        return retrieved_docs, answer
