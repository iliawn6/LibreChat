"""
Configuration constants and paths for the LibreChat RAG project.

All core settings (paths, chunking parameters, model names) are centralized
here so they are easy to locate and adjust. The module is intentionally small
and free of side effects so it can be safely imported from anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, List


#: Project root directory (folder containing this package and project files).
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Data locations
# ---------------------------------------------------------------------------

#: Base directory that contains the raw data assets used by the project.
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

#: Path to the Persian PDF book used as one of the primary data sources.
PDF_PATH: Final[Path] = DATA_DIR / "justforfun_persian.pdf"

#: Directory containing pre-downloaded Stallman HTML pages.
HTML_DIR: Final[Path] = DATA_DIR / "html"

#: URL of the “Linux and Life” book that is loaded directly from the web.
WEB_BOOK_URL: Final[str] = "https://linuxbook.ir/all.html"

#: Directory where the Chroma vector store is persisted.
COLLECTION_DIR: Final[Path] = PROJECT_ROOT / "collection"

#: Name of the Chroma collection used for this project.
CHROMA_COLLECTION_NAME: Final[str] = "Linux_Philosophy"

# ---------------------------------------------------------------------------
# Text splitting configuration
# ---------------------------------------------------------------------------

#: Maximum number of characters per text chunk passed to the embedding model.
CHUNK_SIZE: Final[int] = 700

#: Number of overlapping characters between consecutive chunks.
CHUNK_OVERLAP: Final[int] = 100

# ---------------------------------------------------------------------------
# Embedding and vector store configuration
# ---------------------------------------------------------------------------

#: Batch size used when adding documents to the vector store.
EMBED_BATCH_SIZE: Final[int] = 8

#: Sleep (in seconds) between embedding batches to avoid rate limits.
EMBED_SLEEP_SECONDS: Final[float] = 3.0

#: Name of the Cohere embedding model (default; preferred for best quality).
COHERE_EMBED_MODEL_NAME: Final[str] = "embed-multilingual-v3.0"

#: HuggingFace BGE model for optional local embeddings (requires sentence-transformers, torch).
HF_BGE_MODEL_NAME: Final[str] = "BAAI/bge-m3"

#: Chat model used for RAG question answering.
COHERE_CHAT_MODEL_NAME: Final[str] = "command-r-plus"

# ---------------------------------------------------------------------------
# Wikipedia configuration
# ---------------------------------------------------------------------------

#: List of Persian Wikipedia pages used as additional knowledge sources.
WIKIPEDIA_TITLES: Final[List[str]] = [
    "ریچارد استالمن",
    "لینوس توروالدز",
    "لینوکس",
    "پروژه گنو",
    "نرم‌افزار آزاد",
    "بنیاد نرم‌افزار آزاد",
]


def resolve_project_path(*parts: str) -> Path:
    """
    Resolve a path relative to :data:`PROJECT_ROOT`.

    This helper is convenient when you need to construct paths in scripts or
    notebooks without hard-coding absolute locations.
    """

    return PROJECT_ROOT.joinpath(*parts)

