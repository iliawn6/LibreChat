"""
Text preprocessing utilities for the LibreChat RAG project.

This module contains the Persian text normalization logic used to clean
artifacts produced by PDF extraction while preserving information relevant
for RAG.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

from langchain_core.documents import Document

# Bidirectional control characters that should be stripped from the text.
BIDI_CONTROL = {
    "\u200e",  # LRM
    "\u200f",  # RLM
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",  # embeddings/overrides
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",  # isolates
}

# Zero-width non-joiner (نیم‌فاصله) kept for Persian morphology.
ZWNJ = "\u200c"


def normalize_persian_text(text: str) -> str:
    """
    Apply basic normalization to Persian text extracted from PDFs:

    - Normalize Unicode form (NFKC).
    - Remove bidi control characters but keep ZWNJ.
    - Drop combining marks, which often cause artifacts such as ``ͬ`` or ``ͷ``.
    - Unify Arabic and Persian variants of common letters.
    - Normalize whitespace and newlines.
    - Re-introduce ZWNJ for common patterns such as ``می`` / ``نمی`` prefixes
      and plural forms ending with ``ها`` / ``های``.
    """

    # 1) Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # 1.5) FIRST-STEP GLYPH FIXES (explicit rules from the notebook)
    text = text.replace("مͬ", "می")
    text = text.replace("ͷ", "ک")

    # 2) Remove bidi control chars (keep ZWNJ)
    text = "".join(ch for ch in text if ch not in BIDI_CONTROL)

    # 3) Remove combining marks (category Mn)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # 4) Unify Arabic and Persian variants of common letters
    text = text.replace("\u064a", "\u06cc")  # Arabic ye -> Persian ye
    text = text.replace("\u0643", "\u06a9")  # Arabic ke -> Persian ke

    # 5) Normalize whitespace while preserving Persian sentence flow
    text = re.sub(r"[ \t]+", " ", text)  # collapse consecutive spaces
    text = re.sub(r"\n{2,}", "\n", text)  # collapse multiple blank lines
    text = re.sub(r" *\n *", "\n", text)  # trim spaces around newlines

    # 6.1) Prefixes: می / نمی ---
    # "می " + verb -> "می‌"
    # "نمی " + verb -> "نمی‌"
    text = re.sub(r"\b(ن?می)\s+", rf"\1{ZWNJ}", text)

    # 6.2) Plural "ها" ---
    # "کفش ها" -> "کفش‌ها"
    text = re.sub(
        r"([آ-ی0-9A-Za-z])\s+(هایی|های|ها)\b",
        rf"\1{ZWNJ}\2",
        text,
    )

    return text.strip()


def normalize_documents(docs: Iterable[Document]) -> List[Document]:
    """
    Normalize the ``page_content`` of a collection of :class:`Document` objects.

    The documents are updated in place, and the same list is returned for
    convenience.
    """

    normalized_docs = []
    for doc in docs:
        if isinstance(doc.page_content, str):
            doc.page_content = normalize_persian_text(doc.page_content)
        normalized_docs.append(doc)
    return normalized_docs

