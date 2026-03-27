# Weeks 3 and 4 - RAG Systems Project Summary

This repository documents the Weeks 3 and 4 guided lab work from the AI Engineer Essentials course by Zuu Crew.

The project starts with a minimal RAG pipeline built from scratch, then moves into framework-based PDF RAG, conversational memory and LCEL, hybrid retrieval with reranking, and finally multimodal retrieval with CLIP on dermatology images. Across the repo, the core theme is the same: retrieval quality depends on data preparation, chunking, embeddings, ranking, and how cleanly the system is wired together.

## What I Learned

### 1. RAG fundamentals from first principles

The notebook [notebooks/01_naive_rag_from_scratch_template.ipynb](notebooks/01_naive_rag_from_scratch_template.ipynb) builds a basic RAG system without relying on a framework:

- Load text files from `data/raw_text/`
- Split them into overlapping chunks
- Convert chunks into TF-IDF vectors with `TfidfVectorizer`
- Retrieve relevant chunks with cosine similarity
- Build a prompt from the retrieved context
- Generate a grounded answer with an LLM

Key takeaways:

- RAG is fundamentally `chunk -> represent -> retrieve -> generate`
- TF-IDF is simple and useful for understanding retrieval mechanics
- Overlap matters because context can be split across chunk boundaries
- Prompting with retrieved context makes LLM answers more grounded
- Sparse retrieval has clear limits because it depends heavily on keyword overlap

This notebook also persists useful artifacts to `artifacts/naive_tfidf/`:

- `vectorizer.pkl`
- `matrix.npz`
- `chunks.parquet`

### 2. Framework-based PDF RAG with LangChain

The notebook [notebooks/02_langchain_pdf_rag_chunking_template.ipynb](notebooks/02_langchain_pdf_rag_chunking_template.ipynb) moves from a manual RAG pipeline to a more production-style workflow using LangChain and ChromaDB.

Main topics:

- Loading PDFs with `PyPDFLoader`
- Splitting documents with LangChain text splitters
- Comparing chunking strategies
- Embedding chunks with a configurable embedding model
- Persisting vector stores in Chroma
- Querying the store through a `RetrievalQA` chain

Key takeaways:

- Chunking strategy strongly affects retrieval quality
- Recursive chunking is often better when document structure needs to be preserved
- Fixed-size chunking is easier to reason about and reproduce
- Vector databases are not just storage; they are part of the retrieval design
- Persisted manifests are useful for tracking how an index was built

The current repo already contains two generated LangChain collections:

- `langchain_recursive`: 280 chunks
- `langchain_fixed`: 275 chunks

Those values come from the manifests in `artifacts/manifests/` and reflect the current build settings:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- embedding provider: `sbert`
- chunk size: `800`
- chunk overlap: `150`
- loaded PDF pages/documents: `198`

### 3. Conversational memory and LCEL

The notebook [notebooks/03_memory_lcel_basics_template.ipynb](notebooks/03_memory_lcel_basics_template.ipynb) shifts away from retrieval and focuses on conversation flow and chain composition.

Memory concepts covered:

- `ConversationBufferMemory` keeps the full conversation history
- `ConversationSummaryMemory` compresses history with the help of an LLM
- Buffer memory preserves details but grows without bound
- Summary memory is cheaper over long conversations but may lose precision
- Memory belongs to product behavior, not just model behavior

LCEL concepts covered:

- LangChain components can be composed declaratively with the `|` operator
- `RunnablePassthrough` is useful when input needs to be forwarded unchanged
- `RunnableParallel` helps assemble parallel inputs cleanly
- Streaming improves UX by reducing full-response wait time
- `.with_retry()` and fallbacks are practical resiliency patterns

This notebook also includes a minimal tool-use example and saves a manifest to `artifacts/manifests/memory_lcel.json`.

### 4. Advanced retrieval: dense, sparse, fusion, reranking

The notebook [notebooks/04_advanced_retrieval_rerank_finetune_template.ipynb](notebooks/04_advanced_retrieval_rerank_finetune_template.ipynb) shows that retrieval is not a single technique. Different ranking methods capture different relevance signals.

Main topics:

- Dense retrieval with Chroma vector search
- Sparse retrieval with `BM25Okapi`
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- An end-to-end hybrid RAG pipeline

Key takeaways:

- Dense retrieval is good for semantic similarity
- BM25 is strong when exact words matter
- Hybrid retrieval works because dense and sparse methods fail in different ways
- Fusion improves candidate recall
- Reranking improves final precision, but adds latency and cost

The retrieval pipeline used here is:

```text
Query
  -> Dense retrieval
  -> BM25 retrieval
  -> RRF fusion
  -> Cross-encoder reranking
  -> Final top-k context
  -> LLM answer
```

This notebook also builds a persistent dense collection in `artifacts/chroma/advanced_dense/`.

### 5. Multimodal RAG with CLIP

The notebook [notebooks/06_multimodal_rag_health_clip.ipynb](notebooks/06_multimodal_rag_health_clip.ipynb) explores how retrieval changes when the corpus includes images as well as text.

Main topics:

- Load a subset of DermaMNIST / HAM10000 dermatology images
- Embed images with CLIP
- Store text descriptions and image embeddings separately
- Support text-only retrieval
- Support image retrieval driven by text queries
- Fuse text and image scores into a single ranked result set
- Generate an answer from multimodal context

Key takeaways:

- CLIP puts text and images into compatible embedding spaces
- Multimodal retrieval expands what "relevance" means
- Text retrieval and image retrieval can disagree, so score fusion matters
- Visualization is useful for debugging retrieval quality
- In healthcare-style demos, safety and scope disclaimers matter

The config currently sets `subset_n: 50`, so this notebook works on a manageable sample of 50 extracted images. The notebook defines seven dermatology categories:

- Actinic Keratoses
- Basal Cell Carcinoma
- Benign Keratosis
- Dermatofibroma
- Melanoma
- Melanocytic Nevi
- Vascular Lesions

This is an educational multimodal search demo, not a medical system.

## Reusable Architecture In `src`

The notebooks are the learning path, but the real engineering lesson is in [src/services/llm_services.py](src/services/llm_services.py) and [src/config/config.yaml](src/config/config.yaml).

### Centralized configuration matters

`src/config/config.yaml` acts as a single source of truth for:

- LLM provider and model selection
- OpenRouter settings
- embedding provider and model
- chunk size and overlap
- retrieval settings such as `similarity_top_k`
- hybrid retrieval weights
- CLIP and multimodal settings
- artifact and data paths

This structure makes the notebook workflow much easier to maintain because experimental settings are not hardcoded repeatedly across notebooks.

### Factory functions reduce duplication

`src/services/llm_services.py` contains reusable factories for:

- LangChain chat models
- LangChain text embeddings
- CLIP models
- LlamaIndex LLMs
- LlamaIndex embeddings
- config loading
- API key validation

This file demonstrates a cleaner experimentation layer:

- switch providers without rewriting notebook logic
- keep model initialization consistent
- support local and hosted embeddings behind one interface
- separate configuration concerns from notebook teaching logic

The file currently supports multiple providers and integrations, including:

- OpenAI
- OpenRouter
- Groq
- Gemini
- Ollama
- SBERT / HuggingFace embeddings
- Cohere embeddings
- CLIP through `sentence-transformers`
- LlamaIndex adapters

### Utility scripts make the repo easier to work with

[src/scripts/verify_install.py](src/scripts/verify_install.py) shows the value of verifying an environment instead of assuming it works. It checks imports across the project stack and gives a fast signal that dependencies are installed correctly.

[src/scripts/update_notebook_imports.py](src/scripts/update_notebook_imports.py) shows how repo refactors can be automated instead of patched manually notebook by notebook.

## Skills and Technical Patterns

This repo covers:

- document chunking with overlap
- sparse retrieval with TF-IDF and BM25
- dense retrieval with embedding models
- Chroma vector store creation and persistence
- retrieval quality comparison through chunking strategies
- prompt construction for grounded answering
- conversational memory patterns
- LCEL composition and streaming
- retry and fallback design
- hybrid retrieval with RRF
- reranking with cross-encoders
- multimodal text-image retrieval with CLIP
- centralized config and reusable service factories

## Notebook Order

Recommended notebook order:

1. [notebooks/01_naive_rag_from_scratch_template.ipynb](notebooks/01_naive_rag_from_scratch_template.ipynb)
2. [notebooks/02_langchain_pdf_rag_chunking_template.ipynb](notebooks/02_langchain_pdf_rag_chunking_template.ipynb)
3. [notebooks/03_memory_lcel_basics_template.ipynb](notebooks/03_memory_lcel_basics_template.ipynb)
4. [notebooks/04_advanced_retrieval_rerank_finetune_template.ipynb](notebooks/04_advanced_retrieval_rerank_finetune_template.ipynb)
5. [notebooks/06_multimodal_rag_health_clip.ipynb](notebooks/06_multimodal_rag_health_clip.ipynb)

## Project Structure

```text
Week 03/
|-- notebooks/
|   |-- 01_naive_rag_from_scratch_template.ipynb
|   |-- 02_langchain_pdf_rag_chunking_template.ipynb
|   |-- 03_memory_lcel_basics_template.ipynb
|   |-- 04_advanced_retrieval_rerank_finetune_template.ipynb
|   `-- 06_multimodal_rag_health_clip.ipynb
|-- src/
|   |-- config/
|   |   `-- config.yaml
|   |-- scripts/
|   |   |-- verify_install.py
|   |   `-- update_notebook_imports.py
|   `-- services/
|       `-- llm_services.py
|-- artifacts/
|   |-- chroma/
|   |-- manifests/
|   `-- naive_tfidf/
|-- data_hide/
|   |-- raw_text/
|   |-- pdfs/
|   `-- images/
|-- Theoretical_Guide_RAG_Systems.pdf
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Final Takeaways

The main takeaway from this repo is that RAG quality does not come from "adding a vector database" by itself. Strong RAG systems depend on how documents are chunked, how they are embedded, how results are retrieved and reranked, how prompts are built, and how the system is structured for repeatable experimentation.

## Credits

This project was completed as part of the AI Engineer Essentials course by Zuu Crew, with guidance from the tutor. The README summarizes the main concepts, methods, and technical patterns learned through that guided coursework.
