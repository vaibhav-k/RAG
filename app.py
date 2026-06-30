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

            for i, doc in enumerate(docs, start=1):

                print(
                    f"{i}. "
                    f"{doc['document']['title']} "
                    f"(score={doc['score']:.3f})"
                )

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
