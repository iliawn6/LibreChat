"""
RAG chain construction for the LibreChat project.

This module exposes helpers to build a retriever-backed chat chain using
LangChain and Cohere.
"""

from __future__ import annotations

from typing import Any

from langchain_cohere import ChatCohere
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from .config import COHERE_CHAT_MODEL_NAME


def build_retriever(vector_store: Chroma) -> BaseRetriever:
    """
    Create an MMR-based retriever over the given vector store.

    The parameters mirror the notebook implementation:

    - ``search_type = \"mmr\"``
    - ``fetch_k = 50``
    - ``k = 10``
    """

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"fetch_k": 50, "k": 10},
    )


def _format_docs(docs: list[Document]) -> str:
    """Join retrieved documents into a single string separated by markers."""

    return "\n\n-----\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever: BaseRetriever) -> Runnable:
    """
    Construct the RAG chain used for question answering.

    The chain:

    - Retrieves context documents using the provided retriever.
    - Formats them into a single context string.
    - Feeds the context and the question into a chat model with a Persian
      system prompt.
    - Parses the output as plain text.
    """

    system_prompt = (
        "You are a helpful assistant answering questions about Linux, free software, "
        "and related topics using the provided context. "
        "Always answer in Persian, be concise, and the answers must be short (<= 4 words). "
        "If a query expects only a number, return only the number (no percentage sign or extras). "
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "سوال:\n{question}\n\n"
                "متن‌های بازیابی‌شده:\n{context}\n\n"
                "پاسخ کوتاه را فقط بر اساس متن‌های بالا تولید کن.",
            ),
        ]
    )

    llm = ChatCohere(model=COHERE_CHAT_MODEL_NAME)

    chain: Runnable = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def answer_question(chain: Runnable, question: str) -> str:
    """
    Convenience wrapper around ``chain.invoke`` for a single question.

    Parameters
    ----------
    chain:
        A runnable RAG chain as returned by :func:`build_rag_chain`.
    question:
        The natural language question to answer.
    """

    result = chain.invoke(question)
    return str(result)

