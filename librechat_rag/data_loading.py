"""
Data loading utilities for the LibreChat RAG project.

This module contains small, focused functions that load the different document
sources used by the system:

- A Persian PDF book (``justforfun_persian.pdf``)
- The \"Linux and Life\" web book
- A set of Persian Wikipedia pages
- Local HTML pages mirrored from ``https://stallman.org``

Each loader returns a list of LangChain :class:`~langchain_core.documents.Document`
instances, which can later be preprocessed, split, and embedded.
"""

from __future__ import annotations

from typing import List

from langchain_community.document_loaders import (
    BSHTMLLoader,
    DirectoryLoader,
    PyPDFium2Loader,
    WebBaseLoader,
    WikipediaLoader,
)
from langchain_core.documents import Document

from .config import HTML_DIR, PDF_PATH, WEB_BOOK_URL, WIKIPEDIA_TITLES


def load_pdf_docs() -> List[Document]:
    """
    Load the Persian PDF book as a list of :class:`Document` objects,
    using :class:`~langchain_community.document_loaders.PyPDFium2Loader`
    for robust extraction quality.
    """

    loader = PyPDFium2Loader(str(PDF_PATH))
    docs = loader.load()
    print(f"[data_loading] Loaded {len(docs)} PDF pages from {PDF_PATH}")
    return docs


def load_web_docs() -> List[Document]:
    """
    Load the \"Linux and Life\" web book using :class:`WebBaseLoader`.

    The URL is defined in :mod:`librechat_rag.config`. The loader returns a
    list with one or more :class:`Document` objects depending on how the
    underlying library parses the page.
    """

    loader = WebBaseLoader(WEB_BOOK_URL)
    docs = loader.load()
    print(f"[data_loading] Loaded {len(docs)} web document(s) from {WEB_BOOK_URL}")
    return docs


def load_wiki_docs() -> List[Document]:
    """
    Load a curated set of Persian Wikipedia pages.

    The list of titles is provided by :data:`librechat_rag.config.WIKIPEDIA_TITLES`.
    To avoid hitting Wikipedia API rate limits, callers should add small sleeps
    between subsequent invocations if running many loads in a row.
    """

    all_docs = []
    for title in WIKIPEDIA_TITLES:
        loader = WikipediaLoader(query=title, load_max_docs=1, lang="fa")
        docs = loader.load()
        print(f"[data_loading] Loaded {len(docs)} Wikipedia document(s) for '{title}'")
        all_docs.extend(docs)
    print(f"[data_loading] Total Wikipedia documents: {len(all_docs)}")
    return all_docs


def load_html_docs() -> List[Document]:
    """
    Load local HTML pages mirrored from ``https://stallman.org``.

    The HTML files are expected to live under :data:`librechat_rag.config.HTML_DIR`.
    We use :class:`DirectoryLoader` with :class:`BSHTMLLoader` to extract text
    content from all ``*.html`` files recursively.
    """

    loader = DirectoryLoader(
        str(HTML_DIR),
        glob="**/*.html",
        loader_cls=BSHTMLLoader,
    )
    docs = loader.load()
    print(f"[data_loading] Loaded {len(docs)} HTML document(s) from {HTML_DIR}")
    return docs

