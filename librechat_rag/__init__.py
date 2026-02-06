"""
Top-level package for the LibreChat RAG project.

This package provides a modular implementation of a RAG pipeline for questions
about Linux, free software, and related topics. It is designed to be clean,
reusable, and suitable as a portfolio-quality codebase.

Typical usage from Python:

    from librechat_rag import config, data_loading, preprocessing, splitting
    from librechat_rag import vector_store, rag_chain, evaluation

    # Load or build resources here ...
"""

from . import config, data_loading, preprocessing, splitting, vector_store, rag_chain, evaluation

__all__ = [
    "config",
    "data_loading",
    "preprocessing",
    "splitting",
    "vector_store",
    "rag_chain",
    "evaluation",
]

