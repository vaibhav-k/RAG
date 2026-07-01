"""
Command-line interface for the RAG application.

This module serves as the application's entry point. It initializes
the RAG pipeline, accepts user questions from the terminal, displays
retrieved documents, and prints the generated answers.

Users can exit the application by entering the 'exit' command.
"""

from rag import RAGPipeline


def main():
    rag = RAGPipeline()

    print("=" * 60)
    print(" Amazon RAG Assistant ")
    print("=" * 60)

    while True:

        question = input("\nUser: ")

        if question.lower() == "exit":
            break

        if not question.strip():
            continue

        docs, answer = rag.answer(question)

        if docs:

            print("\nRetrieved Documents")

            for doc in docs:
                title = doc["document"].get("title", "N/A")
                source = doc["document"].get("metadata", {}).get("source", "unknown")
                score = doc.get("score", 0)

                print(f"{title} (score = {score:.3f}) (source file = {source})")

        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
