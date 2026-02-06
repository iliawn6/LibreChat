"""
Document splitting utilities for the LibreChat RAG project.

This module uses :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`
to break long documents into overlapping chunks before embedding.
"""

from __future__ import annotations

from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a :class:`RecursiveCharacterTextSplitter` with project defaults.

    - ``chunk_size = 700``
    - ``chunk_overlap = 100``
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def split_documents(docs: Iterable[Document]) -> List[Document]:
    """
    Split a collection of documents into smaller overlapping chunks.

    Parameters
    ----------
    docs:
        Any iterable of :class:`Document` instances.

    Returns
    -------
    list[Document]
        A new list containing the split document chunks.
    """

    splitter = create_text_splitter()
    doc_list = list(docs)
    splitted = splitter.split_documents(doc_list)
    print(f"[splitting] Split {len(doc_list)} documents into {len(splitted)} chunks")
    return splitted


def preview_split_example(docs: Iterable[Document], max_docs: int = 3) -> None:
    """
    Run a lightweight, non-exhaustive split on a small subset of documents.

    This helper is useful when you want to sanity-check the splitter behavior
    without paying the full cost of splitting the entire corpus.
    """

    subset = list(docs)[:max_docs]
    if not subset:
        print("[splitting] No documents provided for preview.")
        return

    splitter = create_text_splitter()
    chunks = splitter.split_documents(subset)
    print(
        f"[splitting] Preview split: {len(subset)} documents -> {len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )

