## LibreChat RAG (Modular Python Project)

This project implements a modular Retrieval-Augmented Generation (RAG) system
that answers questions about:

- The philosophy of Linux
- Free and open-source software
- Key figures such as Richard Stallman and Linus Torvalds

---

### Project structure

- `librechat_rag/`
  - `__init__.py`: Package initializer; re-exports the main modules.
  - `config.py`: Central configuration (paths, chunking parameters, model names).
  - `data_loading.py`: Load documents from PDF, web, Wikipedia, and local HTML.
  - `html_scraper.py`: Optional utility to scrape and cache HTML pages from `https://stallman.org`.
  - `preprocessing.py`: Persian text normalization utilities.
  - `splitting.py`: Document splitting into overlapping chunks.
  - `vector_store.py`: Cohere embeddings (default) and optional local HuggingFace BGE; Chroma vector store helpers.
  - `rag_chain.py`: Retriever, prompt, and LangChain/Cohere RAG chain.
  - `evaluation.py`: Evaluation questions, answer wrapper, and JSON export.
- `main.py`: Lightweight CLI that runs a small demo using an existing vector store.

The data folders (`data/`, `collection/`, etc.) and any `requirements.txt`
file are expected to be provided by the user/environment.

---

### Data sources

The system uses several complementary sources:

- **PDF book**: `data/justforfun_persian.pdf` (Persian translation of “Just for Fun”).
- **Web book**: Linux and Life (`https://linuxbook.ir/all.html`).
- **Wikipedia pages** (in Persian):
  - ریچارد استالمن
  - لینوس توروالدز
  - لینوکس
  - پروژه گنو
  - نرم‌افزار آزاد
  - بنیاد نرم‌افزار آزاد
- **Local HTML pages**: Mirrored content from `https://stallman.org` under `data/html/`.

These sources are loaded and processed into a single vector store for semantic
retrieval.

---

### Setup

- **Python**: Use a recent Python 3 version compatible with your environment
  (e.g. 3.10).
- **Dependencies**: Install the required packages, including:
  - `langchain`
  - `langchain-core`
  - `langchain-community`
  - `langchain-text-splitters`
  - `langchain-cohere`
  - `chromadb`
  - `cohere`
  - `beautifulsoup4` (for HTML)
  - `pypdfium2`

  You can either reuse your existing virtual environment or add these to your
  current `requirements.txt`.

- **Embedding model**: We **prefer Cohere** for best quality. Set the API key:

  ```bash
  export COHERE_API_KEY="your-api-key-here"
  ```

  **Optional local model**: If you prefer to run embeddings locally (no API key,
  no rate limits), you can use the HuggingFace BGE-m3 model. Install
  `sentence-transformers` and `torch`, then pass
  `vector_store.get_huggingface_embedding_model()` to `build_vector_store` and
  `load_vector_store`. See the full pipeline example below.

- **Cohere credentials** (for chat model and default embeddings):

  The chat model and default embedding model use Cohere; the key must be
  available in the environment before running the project.

---

### Usage

#### 1. Quick connectivity demo (recommended)

If you already have a persisted Chroma collection (for example, created by
calling `build_vector_store` once), you can run:

```bash
python main.py
```

This will:

- Load the existing vector store from `collection/`.
- Build the retriever and RAG chain.
- Ask a single demo question (“ریچارد استالمن کیست؟”) and print the answer as
  a small dictionary.

This path is **lightweight** and does not re-chunk or re-embed the corpus.

---

#### 2. Full pipeline (expensive; optional)

When you want to run the full RAG pipeline in code, you can drive it from a
separate script or a notebook:

```python
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

# 2) Normalize Persian text (typically for PDF pages)
preprocessing.normalize_documents(pdf_docs)

# 3) Split all documents into chunks
all_docs = pdf_docs + wiki_docs + html_docs + web_docs
splitted_docs = splitting.split_documents(all_docs)

# 4) Build a Chroma vector store (API- and time-intensive)
vs = vector_store.build_vector_store(splitted_docs)

# 5) Build the RAG chain and run the 16-question evaluation
retriever = rag_chain.build_retriever(vs)
chain = rag_chain.build_rag_chain(retriever)
answers = evaluation.run_evaluation(chain)

# 6) Save evaluation results to JSON (no zipping)
evaluation.save_answers(answers, "answers.json")
```

This produces a single `answers.json` file suitable for automatic grading
or downstream analysis.

**Using the optional local HuggingFace BGE model** (no Cohere API key needed
for embeddings):

```python
from librechat_rag import vector_store, rag_chain, evaluation

# Use local BGE embeddings instead of Cohere
embedding_model = vector_store.get_huggingface_embedding_model()
vs = vector_store.build_vector_store(splitted_docs, embedding_model=embedding_model)
# ... or to load an existing collection built with BGE:
# vs = vector_store.load_vector_store(embedding_model=embedding_model)
retriever = rag_chain.build_retriever(vs)
chain = rag_chain.build_rag_chain(retriever)
# ... rest of pipeline
```

We prefer Cohere for best quality; the local BGE model is provided as an
optional alternative when you need to run embeddings without an API key.

---

### Implementation notes

- **Modular design**: Each module has a single responsibility (data loading,
  preprocessing, splitting, vector storage, retrieval, evaluation) to keep the
  codebase easy to navigate and extend.
- **No heavy work on import**: All modules are safe to import; expensive
  operations (chunking, embedding, evaluation) are performed only when you
  call the corresponding functions.

