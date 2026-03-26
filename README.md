# RAG Systems - Complete Learning Guide

> A production-grade educational project teaching **Retrieval-Augmented Generation (RAG)** from fundamentals to advanced techniques using real dermatology domain data.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![Framework: LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-orange.svg)](https://docs.llamaindex.ai/)

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Navigate to project directory
cd "RAG Systems"

# 2. Install dependencies (using uv - recommended)
uv venv && source .venv/bin/activate
uv sync

# 3. Verify installation
python src/scripts/verify_install.py  # Should show: ✅ Working: 36/36

# 4. Configure API key (OpenRouter gives access to 100+ models)
# Create .env file and add: OPENROUTER_API_KEY=your_key_here
# Get free key at: https://openrouter.ai/keys

# 5. Start learning!
jupyter lab
# Open: notebooks/01_naive_rag_from_scratch.ipynb (start here!)
```

**That's it!** You're ready to learn RAG systems. 🎉

---

## 📚 Notebooks (Learning Order)

| # | Notebook | Topics | Duration |
|---|----------|--------|----------|
| 01 | `01_naive_rag_from_scratch.ipynb` | TF-IDF, cosine similarity, basic RAG | 30 min |
| 02 | `02_langchain_pdf_rag_chunking.ipynb` | LangChain, ChromaDB, chunking strategies | 45 min |
| 03 | `03_memory_lcel_basics.ipynb` | Memory & LCEL fundamentals | 60 min |
| 04 | `04_advanced_retrieval_rerank_finetune.ipynb` | Hybrid retrieval, reranking, finetuning | 75 min |

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- At least 8GB RAM (16GB recommended for notebook 30)
- ~2GB disk space for models and artifacts

### 2. Installation

Each notebook has its own dependency cell. Uncomment and run:

```bash
# Example for notebook 01
%pip install --quiet pandas numpy scikit-learn \
    langchain langchain-openai langchain-groq \
    python-dotenv
```

**OR** install all dependencies at once:

```bash
pip install pandas numpy pillow matplotlib scikit-learn \
    sentence-transformers chromadb rank-bm25 \
    langchain langchain-openai langchain-groq langchain-google-genai \
    langchain-cohere langchain-community langchain-chroma \
    llama-index llama-index-llms-openai llama-index-embeddings-huggingface \
    llama-index-vector-stores-chroma pypdf python-dotenv
```

### 3. API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Required keys (depending on which providers you use):
- `OPENAI_API_KEY` - For GPT models (most notebooks)
- `GROQ_API_KEY` - For fast Llama inference (optional)
- `GOOGLE_API_KEY` - For Gemini models (optional)
- `COHERE_API_KEY` - For Cohere embeddings/reranking (optional)

### 4. Run Notebooks

Open Jupyter Lab/Notebook and start with `01_naive_rag_from_scratch.ipynb`:

```bash
jupyter lab
```

## 🎯 Learning Path

### Core Learning Track (Notebooks 01-04)
Complete these notebooks in order:
1. **01**: Understand core RAG mechanics without frameworks
2. **02**: Learn LangChain's document loaders and text splitters
3. **03**: Master conversational memory and LCEL composition ⭐
4. **04**: Hybrid retrieval, reranking, and embedding finetuning

## 📂 Project Structure

```
RAG Systems/
├── notebooks/            # 📓 Core learning notebooks (4)
│   ├── 01_naive_rag_from_scratch.ipynb
│   ├── 02_langchain_pdf_rag_chunking.ipynb
│   ├── 03_memory_lcel_basics.ipynb
│   └── 04_advanced_retrieval_rerank_finetune.ipynb
│
├── src/                  # 🏗️  Source code
│   ├── config/
│   │   └── config.yaml   # Centralized configuration
│   ├── services/
│   │   └── llm_services.py  # LLM & embedding factories
│   └── scripts/
│       └── verify_install.py  # Dependency verification
│
├── data/                 # 📊 Input data
│   ├── raw_text/         # Text documents
│   ├── pdfs/             # PDF documents
│   └── images/           # Images for multimodal RAG
│
├── artifacts/            # 🗄️  Generated outputs
│   ├── naive_tfidf/      # TF-IDF artifacts
│   ├── chroma/           # Vector databases
│   └── manifests/        # Build metadata
│
├── docs/                 # 📚 Documentation
│   ├── QUICKSTART.md
│   ├── INSTALL.md
│   ├── ARCHITECTURE.md
│   ├── ENV_TEMPLATE.md
│   ├── OPENROUTER_PROVIDERS_GUIDE.md
│   ├── changelog.md
│   └── stepplan.md
│
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
└── README.md             # This file
```

## 🔧 Configuration

**All notebooks use centralized configuration** via `src/config/config.yaml` and `src/services/llm_services.py` - no code duplication!

### Quick Model Switch

Edit `src/config/config.yaml` to change models across **all notebooks** at once:

```yaml
# Use OpenRouter (access 100+ models with 1 API key)
llm_provider: "openrouter"
openrouter_provider: "google"      # openai | google | anthropic | meta | mistral
openrouter_model: "gemini-flash-1.5"  # FREE model!

# Embeddings
text_emb_provider: "sbert"
text_emb_model: "sentence-transformers/all-MiniLM-L6-v2"

# RAG Settings
chunk_size: 800
chunk_overlap: 150
similarity_top_k: 3
```

**That's it!** Change 2 lines, all notebooks use the new model. ✨

### Available Free Models

- **Google Gemini Flash 1.5** - Fast, high-quality, FREE
- **Meta Llama 3.2 3B** - Good for testing, FREE

See `docs/OPENROUTER_PROVIDERS_GUIDE.md` for 100+ model options.

## 🔄 Rebuild Triggers

ChromaDB collections need rebuilding when you change:
- **Embedding model** → Rebuild that collection
- **Chunking strategy** → Use a new collection name
- **Dataset** → Re-embed and rebuild

Check manifests in `./artifacts/manifests/` to see build parameters.

## 💡 Key Concepts

### RAG Pipeline
```
Documents → Chunk → Embed → Store → Retrieve → Generate
```

### Chunking Strategies
- **Recursive**: Respects document structure (paragraphs → sentences)
- **Fixed**: Predictable chunk sizes with overlap
- **Sentence**: Preserves semantic boundaries

### Retrieval Methods
- **Dense** (Vector): Semantic similarity via embeddings
- **Sparse** (BM25): Keyword-based matching
- **Hybrid**: Combines both using fusion (weighted or RRF)

### Advanced Techniques
- **Reranking**: Cross-encoder refines top-N results
- **Finetuning**: Adapt embeddings to your domain
- **Multimodal**: Combine text and image retrieval

## 💡 Learning Tips

### Notebook-Specific Notes

**01 - Naive RAG:**
- Great for understanding fundamentals
- Shows limitations of TF-IDF (keyword-based, no semantics)
- Sets the stage for dense embeddings

**02 - LangChain PDF RAG:**
- Master document loaders and text splitters
- Compare chunking strategies empirically
- Understand vector database operations

**03 - Memory & LCEL:**
- Critical for conversational AI
- LCEL is LangChain's declarative composition pattern
- Focuses on fundamentals, not RAG

**04 - Advanced Retrieval:**
- Hybrid retrieval combines dense + sparse strengths
- Reranking improves precision at cost of latency
- Production-grade patterns

### Common Pitfalls

1. **Large context windows**: Keep chunk size ~800 chars
2. **Missing API keys**: Check `.env` before running
3. **Memory issues**: Close other applications if needed
4. **Slow embeddings**: Use SBERT (local) instead of API models

## 📊 Performance Expectations

### CPU (M1/M2 Mac or Modern Intel)
- Notebooks 01-03: <5 minutes each
- Notebook 12: <10 minutes
- Notebook 20: <15 minutes (with subset_n=50)
- Notebook 30: <20 minutes (with light finetuning)

### GPU (CUDA-enabled)
- 2-3x faster for notebooks 20 and 30
- Minimal difference for notebooks 01-12

## 🏗️ Architecture Overview

```
RAG Systems/
├── notebooks/              # 4 core learning notebooks
│   ├── 01_naive_rag_from_scratch.ipynb
│   ├── 02_langchain_pdf_rag_chunking.ipynb
│   ├── 03_memory_lcel_basics.ipynb
│   └── 04_advanced_retrieval_rerank_finetune.ipynb
│
├── src/                    # Source code
│   ├── config/
│   │   └── config.yaml     # Centralized configuration
│   └── services/
│       └── llm_services.py # LLM & embedding factories
│
├── data/                   # Input data
│   ├── pdfs/              # Dermatology PDFs
│   └── raw_text/          # Text documents
│
├── artifacts/              # Generated outputs
│   ├── chroma/            # Vector store databases
│   └── manifests/         # Build tracking JSONs
│
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Design Principles Applied

✅ **DRY (Don't Repeat Yourself)** - Config defined once, used everywhere  
✅ **Single Source of Truth** - `config.yaml` controls all notebooks  
✅ **Separation of Concerns** - Data, logic, config separated  
✅ **Composition over Inheritance** - Modular retrieval components  
✅ **Explicit over Implicit** - Clear provider switches, rebuild triggers  

See `docs/ARCHITECTURE.md` for detailed design decisions.

---

## 🛠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **"No module named X"** | Run `uv sync` or `uv pip install -r requirements.txt`, then `python src/scripts/verify_install.py` |
| **"API key not found"** | Check `.env` exists, add `OPENROUTER_API_KEY=xxx`, restart kernel |
| **"ChromaDB collection exists"** | Delete `./artifacts/chroma/<name>/` or use different collection name |
| **Slow embeddings** | Use lighter model: `all-MiniLM-L6-v2`, reduce `subset_n` |
| **Out of memory** | Reduce batch size in notebook 30, close other apps |
| **Import errors in Jupyter** | Install kernel: `python -m ipykernel install --user --name=rag-systems` |

### Still Stuck?

1. **Check logs:** Notebooks show detailed error messages
2. **Verify setup:** Run `python src/scripts/verify_install.py`
3. **Review changelog:** See `docs/changelog.md` for technical details
4. **Check config:** Run `python -m src.services.llm_services` to test configuration

## 📝 License & Usage

This educational suite is designed for learning RAG systems. 

- Sample data is synthetic or openly licensed
- Health images in notebook 20 are placeholders (educational only)
- Models (BERT, GPT, etc.) follow their respective licenses

**Academic Use**: ✅ Encouraged  
**Commercial Use**: ⚠️  Check individual model licenses  
**Medical Use**: ❌ NOT intended for diagnosis/treatment

## 🌟 What Makes This Project Special

✨ **Production-grade code** - Not just tutorials, but real-world patterns  
✨ **Zero duplication** - Centralized config, DRY principles throughout  
✨ **Domain-specific** - Real dermatology data, not toy examples  
✨ **Framework comparison** - LangChain vs LlamaIndex side-by-side  
✨ **100+ LLM access** - OpenRouter integration for easy model switching  
✨ **Complete pipeline** - From naive TF-IDF to advanced finetuning  
✨ **Multimodal ready** - CLIP integration for text+image search  
✨ **Fully documented** - Changelog tracks every decision  

---

## 🤝 Contributing

Suggestions and improvements welcome! This is an educational project designed to teach RAG concepts through hands-on examples.

### Ways to Contribute

- 🐛 **Report bugs** - Open an issue with reproducible steps
- 📝 **Improve docs** - Clarify explanations, fix typos
- 💡 **Suggest features** - New retrieval techniques, evaluation metrics
- 🧪 **Add examples** - More domain-specific use cases
- ⭐ **Share feedback** - What worked? What was confusing?

---

## 📝 Homework Assignments (Optional)

### Overview

The `homework/` folder contains 2 optional notebooks for advanced practice:
- **05_llamaindex_pdf_rag_chunking_template.ipynb** - LlamaIndex framework comparison
- **06_multimodal_rag_health_clip_template.ipynb** - Multimodal RAG with images

These are **optional but highly recommended** for deeper understanding.

---

### Homework 05: LlamaIndex PDF RAG

**Prerequisites:** Complete notebooks 01 and 02  
**Time:** ~60 minutes  
**Difficulty:** ⭐⭐⭐ Intermediate

**What you'll learn:**
- Compare LlamaIndex vs LangChain frameworks
- Use LlamaIndex node parsers (similar to text splitters)
- Build query engines with ChromaDB

**Setup:**
No additional setup needed - uses same PDFs from core notebooks.

**To complete:**
1. Navigate to `homework/` folder
2. Open `05_llamaindex_pdf_rag_chunking_template.ipynb`
3. Complete the YOUR CODE HERE sections (3 exercises)

---

### Homework 06: Multimodal RAG with CLIP

**⚠️ IMPORTANT: Educational demo only - NOT for medical diagnosis**

**Prerequisites:** Complete notebooks 01, 02, and 03  
**Time:** ~90 minutes  
**Difficulty:** ⭐⭐⭐⭐ Advanced

**What you'll learn:**
- Use CLIP to embed both images and text into shared vector space
- Build multimodal retrieval (text→image, image→image)
- Implement score fusion strategies (weighted average)
- Visualize search results with matplotlib

---

### 📥 Dataset Setup for Homework 06

**Step 1: Download DermaMNIST Dataset**

The notebook uses **DermaMNIST** - a curated dataset of 10,000 dermatoscopic images from HAM10000.

**Download Instructions:**
1. Visit: https://zenodo.org/records/10519652
2. Download file: `dermamnist_128.npz` (approximately 350MB)
3. Or direct link: https://zenodo.org/records/10519652/files/dermamnist_128.npz

**Step 2: Place Dataset File**

```bash
# Create directory if it doesn't exist
mkdir -p "Week 03/data/images"

# Move downloaded file to:
Week 03/data/images/dermamnist_128.npz
```

**Step 3: Verify Setup**

Your folder structure should look like:
```
Week 03/
└── data/
    └── images/
        └── dermamnist_128.npz  ✅ (350MB)
```

**Step 4: Run the Notebook**

The notebook will automatically:
- Extract 50 images from the 7,007 available
- Create 7 disease category labels with medical descriptions
- Save individual images to `data/images/dermamnist/`
- Build text and image embeddings with CLIP

---

### 🏥 About DermaMNIST Dataset

**Source:** Subset of HAM10000 dermatology image collection  
**Images:** 7,007 training images (128×128 RGB)  
**Categories:** 7 skin disease types:
1. **Actinic Keratoses** - Pre-cancerous UV damage lesions
2. **Basal Cell Carcinoma** - Most common skin cancer
3. **Benign Keratosis** - Non-cancerous skin growths
4. **Dermatofibroma** - Benign fibrous tumors
5. **Melanoma** - Dangerous skin cancer
6. **Melanocytic Nevi** - Common moles
7. **Vascular Lesions** - Blood vessel growths

**License:** Creative Commons (educational use allowed)  
**Citation:** MedMNIST v2 - https://medmnist.com/

---

### ⚠️ Educational Use Disclaimer

This dataset and notebook are for **EDUCATIONAL PURPOSES ONLY**:
- ❌ NOT for medical diagnosis
- ❌ NOT for clinical decision making
- ❌ NOT a substitute for professional medical advice
- ✅ For learning AI/ML concepts with real medical imagery
- ✅ For understanding multimodal retrieval techniques

---

### 🐛 Troubleshooting

**Issue: "DermaMNIST dataset not found"**
- **Solution:** Download dataset and place in `data/images/dermamnist_128.npz`

**Issue: "Out of memory"**
- **Solution:** Reduce `subset_n` in config.yaml (default: 50, try: 20)

**Issue: "CLIP model download slow"**
- **Solution:** First run downloads model (~350MB), subsequent runs use cached version

**Issue: "ChromaDB collection already exists"**
- **Solution:** Delete `artifacts/chroma/health_*` folders and re-run

---

## 📚 External Resources

### Frameworks
- [LangChain Docs](https://python.langchain.com/) - RAG orchestration
- [LlamaIndex Docs](https://docs.llamaindex.ai/) - Alternative RAG framework
- [ChromaDB Docs](https://docs.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embeddings

### Research Papers
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Multimodal embeddings
- [RAG Survey](https://arxiv.org/abs/2312.10997) - Comprehensive RAG overview
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25) - Sparse retrieval

### LLM Providers
- [OpenRouter](https://openrouter.ai/) - 100+ models, single API
- [OpenAI Platform](https://platform.openai.com/) - GPT models
- [Groq](https://groq.com/) - Fast inference
- [Google AI Studio](https://makersuite.google.com/) - Gemini models

## 🙏 Acknowledgments

Built with:
- LangChain & LlamaIndex for RAG frameworks
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- OpenAI, Groq, Google for LLM APIs

---

**Happy Learning! 🚀**

For questions or issues, please open a GitHub issue or contact the course instructor.

