"""
Document retrieval component.

This module implements semantic search using the FAISS index. Given
a user query, it computes the query embedding, performs similarity
search, filters results based on a similarity threshold, and returns
the most relevant documents.

This module is responsible only for retrieval and does not perform
text generation.
"""

from config import TOP_K, SIMILARITY_THRESHOLD
from data import documents


class Retriever:

    def __init__(self, embedding_index):
        self.embedding_index = embedding_index

    def retrieve(self, query):

        query_embedding = self.embedding_index.embed_query(query)

        scores, indices = self.embedding_index.index.search(query_embedding, TOP_K)

        if scores[0][0] < SIMILARITY_THRESHOLD:
            return None, None

        retrieved_docs = []

        for score, idx in zip(scores[0], indices[0]):

            retrieved_docs.append(
                {
                    "score": float(score),
                    "document": documents[idx],
                }
            )

        return retrieved_docs, scores[0][0]
