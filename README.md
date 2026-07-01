# 📚 Amazon RAG Assistant (Retrieval-Augmented Generation)

A lightweight Retrieval-Augmented Generation (RAG) system built using FAISS and Sentence Transformers to answer Amazon customer support queries using a local knowledge base.

---

## 🚀 Features

- 🔍 Semantic search using SentenceTransformer embeddings
- ⚡ Fast similarity search with FAISS
- 🧠 Context-aware answer generation using FLAN-T5
- 📄 Chunked document support with metadata (title, category, source)
- 🧩 Modular architecture (retriever, embeddings, generator, pipeline)
- 🛠️ Improved prompt design for stable responses

---

## 🏗️ Architecture

```text
User Query
    ↓
Query Embedding (MiniLM)
    ↓
FAISS Vector Search
    ↓
Top-K Relevant Chunks
    ↓
Context Construction (documents + metadata)
    ↓
Prompt Construction
    ↓
FLAN-T5 Generation
    ↓
Final Answer