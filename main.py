"""
Command-line entrypoint for the LibreChat RAG project.

By default this script performs a **lightweight** check of the modular
implementation by:

1. Loading an existing Chroma vector store (no re-embedding).
2. Building the retriever and RAG chain.
3. Asking a single demo question and printing the answer.

This keeps runtime and API usage low while still verifying that the main
components are wired correctly.

For a full evaluation run (which is more expensive), you can import the
package in a separate script or notebook and call:

    from librechat_rag import (
        data_loading,
        preprocessing,
        splitting,
        vector_store,
        rag_chain,
        evaluation,
    )

    # 1) Load raw documents
    pdf_docs = data_loading.load_pdf_docs()
    wiki_docs = data_loading.load_wiki_docs()
    html_docs = data_loading.load_html_docs()
    web_docs = data_loading.load_web_docs()

    # 2) Normalize Persian text where appropriate (e.g. PDF pages)
    preprocessing.normalize_documents(pdf_docs)

    # 3) Split all documents
    all_docs = pdf_docs + wiki_docs + html_docs + web_docs
    splitted_docs = splitting.split_documents(all_docs)

    # 4) Build vector store (expensive)
    vs = vector_store.build_vector_store(splitted_docs)

    # 5) Build chain and run evaluation
    retriever = rag_chain.build_retriever(vs)
    chain = rag_chain.build_rag_chain(retriever)
    answers = evaluation.run_evaluation(chain)
    evaluation.save_answers(answers)
"""

from __future__ import annotations

from librechat_rag import evaluation, rag_chain, vector_store


def main() -> None:
    """
    Run a small end-to-end demo using an existing vector store.

    This function does **not** rebuild the vector store or re-chunk documents;
    it simply checks that:

    - The stored collection can be loaded.
    - The retriever and RAG chain can be constructed.
    - A single demo question can be answered.
    """

    # 1) Load existing vector store (no re-embedding).
    vs = vector_store.load_vector_store()

    # 2) Build retriever and RAG chain.
    retriever = rag_chain.build_retriever(vs)
    chain = rag_chain.build_rag_chain(retriever)

    # 3) Ask a single demo question.
    demo_question = "ریچارد استالمن کیست؟"
    from librechat_rag.rag_chain import answer_question

    answer_text = answer_question(chain, demo_question)
    answer = evaluation.create_QA_result(0, answer_text)

    print("\n[main] Demo question:")
    print(demo_question)
    print("[main] Answer dictionary:")
    print(answer)


if __name__ == "__main__":
    main()

