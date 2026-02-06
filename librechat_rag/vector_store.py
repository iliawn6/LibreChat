"""
Vector store utilities for the LibreChat RAG project.

This module encapsulates the configuration of the Cohere embedding model and
the Chroma vector store. It provides helpers to either build a new collection
from split documents or load an existing one.

**Embedding backends**: Cohere is the default and preferred option. An optional
local HuggingFace BGE model is available via :func:`get_huggingface_embedding_model`
for environments where you prefer to run embeddings locally (no API key required).
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional

from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import (
    CHROMA_COLLECTION_NAME,
    COHERE_EMBED_MODEL_NAME,
    COLLECTION_DIR,
    EMBED_BATCH_SIZE,
    EMBED_SLEEP_SECONDS,
    HF_BGE_MODEL_NAME,
)


def get_embedding_model() -> CohereEmbeddings:
    """
    Create and return the Cohere embedding model used for this project.

    The caller is responsible for ensuring that the ``COHERE_API_KEY``
    environment variable is set before invoking this function.
    """

    return CohereEmbeddings(
        model=COHERE_EMBED_MODEL_NAME,
        max_retries=2,
        request_timeout=40,
    )


def get_huggingface_embedding_model() -> Embeddings:
    """
    Create and return the local HuggingFace BGE embedding model.

    This is an **optional** alternative to Cohere for environments where you
    prefer to run embeddings locally (no API key, no rate limits). Requires
    ``sentence-transformers`` and ``torch``. Uses GPU if available.

    The model (BAAI/bge-m3) is multilingual and suitable for Persian text.
    We prefer Cohere for best quality; use this when local execution is needed.
    """

    try:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    except ImportError as e:
        raise ImportError(
            "HuggingFace embeddings require langchain-community. "
            "For local BGE model, also install: sentence-transformers, torch"
        ) from e

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceBgeEmbeddings(
        model_name=HF_BGE_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    splitted_docs: Iterable[Document],
    embedding_model: Optional[Embeddings] = None,
) -> Chroma:
    """
    Build (or extend) a Chroma vector store from an iterable of split documents.

    This function performs batched insertion with small sleeps between batches
    to reduce the chance of hitting rate limits on the embedding service.

    Parameters
    ----------
    splitted_docs:
        Iterable of document chunks that are ready to be embedded.
    embedding_model:
        Embedding model to use. Defaults to Cohere. Pass
        :func:`get_huggingface_embedding_model` for local BGE embeddings.
    """

    if embedding_model is None:
        embedding_model = get_embedding_model()
    vector_store = Chroma(
        embedding_function=embedding_model,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(COLLECTION_DIR),
    )

    docs_list = list(splitted_docs)
    total = len(docs_list)
    print(
        f"[vector_store] Adding {total} documents to Chroma "
        f"(batch_size={EMBED_BATCH_SIZE})"
    )

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = docs_list[start : start + EMBED_BATCH_SIZE]
        vector_store.add_documents(batch)
        time.sleep(EMBED_SLEEP_SECONDS)

    # Chroma 0.4+ auto-persists, but calling persist() is harmless and explicit.
    vector_store.persist()
    print(f"[vector_store] Collection size: {vector_store._collection.count()}")
    return vector_store


def load_vector_store(
    embedding_model: Optional[Embeddings] = None,
) -> Chroma:
    """
    Load an existing Chroma collection without re-adding documents.

    This is the preferred entry point when you already have a persisted
    collection on disk created by :func:`build_vector_store` or by another
    ingestion script. It performs no heavy recomputation.

    Parameters
    ----------
    embedding_model:
        Embedding model used when the collection was built. Must match the
        model used at build time for correct similarity search. Defaults to
        Cohere. Pass :func:`get_huggingface_embedding_model` if the collection
        was built with the local BGE model.
    """

    if embedding_model is None:
        embedding_model = get_embedding_model()
    vector_store = Chroma(
        embedding_function=embedding_model,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(COLLECTION_DIR),
    )
    print(
        f"[vector_store] Loaded existing collection '{CHROMA_COLLECTION_NAME}' "
        f"from {COLLECTION_DIR} (size={vector_store._collection.count()})"
    )
    return vector_store

