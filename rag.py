"""
Retrieval-Augmented Generation (RAG) pipeline.

This module coordinates the complete RAG workflow by combining
document retrieval and language generation. It retrieves the most
relevant documents for a user query, constructs a prompt using the
retrieved context, and generates an answer using the language model.

The RAGPipeline class provides a single interface for answering
questions.
"""

from config import PROMPT_TEMPLATE
from retriever import Retriever
from generator import Generator
from embeddings import EmbeddingIndex


class RAGPipeline:

    def __init__(self):

        self.embedding_index = EmbeddingIndex()

        self.retriever = Retriever(self.embedding_index)

        self.generator = Generator()

    def answer(self, question):

        retrieved_docs, score = self.retriever.retrieve(question)

        if retrieved_docs is None:

            return (None, "Sorry, I couldn't find a relevant document.")

        context = "\n\n".join(doc["document"]["text"] for doc in retrieved_docs)

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        answer = self.generator.generate(prompt)

        return retrieved_docs, answer
