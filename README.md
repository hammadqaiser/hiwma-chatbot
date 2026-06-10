# HIWMA Waste Management RAG Chatbot

A production-grade, Retrieval-Augmented Generation (RAG) chatbot that answers technical questions about solid waste management using content from 6 engineering textbooks. Built as a modular component of the **HIWMA (Holistic & Integrated Waste Management Approach)** decision-support framework.

The chatbot ingests PDF manuals, builds a searchable vector index, and generates accurate, citation-backed answers grounded in the source literature. It ships as both a standalone web application and an embeddable widget that can be dropped into any existing dashboard or simulator.

---

## Context

HIWMA is an end-to-end decision-support framework that integrates data management, technology selection, and sustainability assessment to generate optimized solid waste management scenarios. The project is a collaboration between [Spect-AI](https://spect.ai), [Lahore University of Management Sciences (LUMS)](https://lums.edu.pk), [Northumbria University](https://www.northumbria.ac.uk/), and the [University of Engineering and Technology, Lahore](https://www.uet.edu.pk/), conducted under the guidance of Muhammad Awais Mian, Talib E. Butt, and Qurat Ul Ain Quraishi.

This chatbot module provides the knowledge retrieval layer — allowing government officials, researchers, and field operators to query waste management literature in natural language and receive technically accurate, source-cited answers.

---

## Architecture

The system is organized into 5 independent, test-driven modules that communicate through well-defined interfaces:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HIWMA RAG CHATBOT                              │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │  MODULE 1    │   │  MODULE 2    │   │  MODULE 3    │                │
│  │  Document    │──▶│  RAG Engine  │──▶│  LLM Gateway │                │
│  │  Ingestion   │   │  (Retriever) │   │  (Model API) │                │
│  └──────────────┘   └──────┬───────┘   └──────┬───────┘                │
│                            │                   │                        │
│                     ┌──────▼───────────────────▼───────┐               │
│                     │         MODULE 4                  │               │
│                     │   Backend API (FastAPI)           │               │
│                     └──────────────┬───────────────────┘               │
│                                    │                                    │
│                     ┌──────────────▼───────────────────┐               │
│                     │         MODULE 5                  │               │
│                     │   Frontend Chat UI                │               │
│                     │   (Standalone + Widget)           │               │
│                     └──────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
User Query ──▶ Casual Detection ──▶ (greeting?) ──▶ Friendly Response (no RAG)
                     │
                     ▼ (knowledge question)
              Hybrid Retrieval ──▶ Context Formatting ──▶ LLM Generation ──▶ Response + Citations
              (Vector + BM25)
```

---

## Project Structure

```
hiwma-chatbot/
├── chatbot/
│   ├── ingestion/            # M1: PDF → chunks → embeddings → ChromaDB
│   │   ├── parser.py         # PyMuPDF PDF extraction with metadata
│   │   ├── chunker.py        # Overlap-aware text chunking
│   │   ├── embedder.py       # Sentence-Transformers local embeddings
│   │   ├── store.py          # ChromaDB vector store (batched upserts)
│   │   └── cli.py            # CLI entry point for ingestion
│   ├── retriever/            # M2: Query → relevant chunks
│   │   ├── base.py           # BaseRetriever interface
│   │   ├── vector_retriever.py
│   │   ├── keyword_retriever.py   # BM25 lexical search
│   │   ├── hybrid_retriever.py    # 70% semantic + 30% keyword
│   │   ├── context_formatter.py   # Formats chunks for LLM consumption
│   │   └── factory.py
│   ├── llm_gateway/          # M3: Context + query → LLM → answer
│   │   ├── base.py           # BaseLLM interface
│   │   ├── prompt_templates.py    # Anti-hallucination system prompts
│   │   ├── groq_llm.py       # Groq (default)
│   │   ├── google_llm.py     # Google Gemini
│   │   ├── openai_llm.py     # OpenAI / Azure
│   │   ├── anthropic_llm.py  # Anthropic Claude
│   │   ├── ollama_llm.py     # Local Ollama
│   │   └── factory.py
│   ├── api/                  # M4: FastAPI REST server
│   │   ├── main.py           # App entry point, static serving, CORS
│   │   ├── models.py         # Pydantic request/response schemas
│   │   ├── chat_history.py   # SQLite session persistence
│   │   ├── routes/
│   │   │   ├── chat.py       # POST /api/chat
│   │   │   ├── health.py     # GET /api/health
│   │   │   └── documents.py  # GET /api/documents
│   │   └── services/
│   │       └── orchestrator.py    # Core pipeline orchestration
│   └── frontend/             # M5: Chat UI
│       ├── index.html        # Standalone chat page
│       ├── styles.css         # Dark glassmorphism design system
│       ├── chat.js           # Chat logic and API client
│       ├── widget.js         # Embeddable Shadow DOM widget
│       └── widget-demo.html  # Widget integration demo
├── tests/                    # 160 automated tests (pytest)
│   ├── ingestion/            # 37 tests
│   ├── retriever/            # 37 tests
│   ├── llm_gateway/          # 44 tests
│   └── api/                  # 42 tests
├── config.yaml               # All pipeline settings (models, chunking, retrieval)
├── .env.example              # Template for API keys
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Build configuration
```

**Not in the repository** (gitignored):
- `Books/` — Source PDF textbooks (~135 MB total). Place your own PDFs here before ingestion.
- `data/` — Generated at runtime. Contains the ChromaDB vector store and SQLite chat history.
- `.env` — Your actual API keys. Never committed.

---

## Modules

### Module 1: Document Ingestion

Parses 6 PDF textbooks (3,765 pages total), splits them into overlapping chunks of 1000 tokens with 200-token overlap, generates 384-dimensional embeddings using `all-MiniLM-L6-v2` (runs locally, no API cost), and stores everything in ChromaDB with metadata (book title, chapter, page number, page range).

Handles large corpora by upserting in batches of 5,000 to stay within ChromaDB's per-call limits.

### Module 2: RAG Retriever

Hybrid search combining vector similarity (70% weight) and BM25 keyword matching (30% weight). The keyword component catches exact technical terms that pure semantic search can miss (e.g., "HDPE geomembrane" or "BOD5"). Results are deduplicated by chunk ID and formatted with full source metadata for citation.

### Module 3: LLM Gateway

Factory-pattern abstraction over 5 LLM providers. All providers implement a common `BaseLLM` interface, so switching from Groq to Gemini to OpenAI to a local Ollama model is a single config change — no code modifications. The system prompt enforces anti-hallucination guardrails: answer only from retrieved context, no inline citations (the UI handles citation display), and graceful fallback when information is missing.

### Module 4: Backend API

FastAPI application that orchestrates the pipeline. The `Orchestrator` class detects casual messages (greetings, thanks) via regex patterns and skips RAG retrieval entirely for those — saving API tokens and reducing latency. Knowledge questions flow through the full retrieve → format → generate pipeline. Chat sessions are persisted in SQLite with async I/O.

### Module 5: Frontend UI

Two deployment modes:
- **Standalone page** — Full-screen dark glassmorphism chat interface with expandable citation cards, session management, and responsive design.
- **Embeddable widget** — A floating chat bubble (`widget.js`) that uses Shadow DOM to guarantee zero CSS conflicts when embedded in an external simulator or dashboard. One line of HTML to integrate.

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.11+ | Best AI/ML ecosystem, async support |
| PDF Parsing | PyMuPDF | Free, fast, handles metadata extraction |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | Free local embeddings, no API cost |
| Vector Database | ChromaDB | Free, file-based, sufficient for target scale |
| Default LLM | Groq (Llama 3.3 70B) | Free tier available, fast inference |
| Alternative LLMs | Google Gemini, OpenAI, Anthropic, Ollama | Swap via config — no code change |
| Backend | FastAPI + Uvicorn | Async, auto-generated docs, production-ready |
| Frontend | Vanilla HTML/CSS/JS | Zero framework dependencies, embeddable anywhere |
| Chat History | SQLite (aiosqlite) | Lightweight, zero-config, file-based |
| Testing | pytest | 160 tests across all modules |

---

## Setup

### Prerequisites

- Python 3.11 or higher
- An API key for at least one LLM provider (Groq free tier recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/hammadqaiser/hiwma-chatbot.git
cd hiwma-chatbot

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy the environment template and add your API key
cp .env.example .env
# Edit .env — set GROQ_API_KEY (or whichever provider you use)
```

The active LLM provider and all pipeline parameters are configured in `config.yaml`:

```yaml
llm:
  provider: "groq"                   # Options: groq, google, openai, anthropic, ollama
  model: "llama-3.3-70b-versatile"
  temperature: 0.1
```

### Data Ingestion (run once)

Place your PDF textbooks in a `Books/` directory at the project root, then run:

```bash
python -m chatbot.ingestion.cli --source ./Books
```

First run downloads the embedding model (~90 MB) and processes all PDFs. The vector index is persisted in `data/chromadb/` and reused on subsequent starts.

### Start the Server

```bash
uvicorn chatbot.api.main:app --reload --host 0.0.0.0 --port 8000
```

| Interface | URL |
|-----------|-----|
| Chat Application | http://localhost:8000/ |
| Widget Demo | http://localhost:8000/widget-demo.html |
| API Documentation (Swagger) | http://localhost:8000/docs |

### Running Tests

```bash
# All 160 tests
python -m pytest tests/ -v

# Per module
python -m pytest tests/ingestion/ -v      # 37 tests
python -m pytest tests/retriever/ -v      # 37 tests
python -m pytest tests/llm_gateway/ -v    # 44 tests
python -m pytest tests/api/ -v            # 42 tests
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send a message. Accepts `{ "message": "...", "session_id": "..." }`. Returns answer, sources, model info, and token usage. |
| `GET` | `/api/documents` | List all ingested documents with chunk counts and page counts. |
| `GET` | `/api/health` | Service health check. Returns status, provider, model, and retriever type. |

### Example Request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is leachate and how is it managed in landfills?"}'
```

### Example Response

```json
{
  "answer": "Leachate is liquid that percolates through waste in a landfill...",
  "sources": [
    {
      "book_title": "Solid Waste Management Engineering",
      "chapter": "Chapter 5 — Landfill Design",
      "page_range": "120-122",
      "relevance_score": 0.92
    }
  ],
  "model": "llama-3.3-70b-versatile",
  "provider": "groq",
  "session_id": "abc123",
  "usage": { "prompt_tokens": 1200, "completion_tokens": 350, "total_tokens": 1550 }
}
```

---

## Switching LLM Providers

| Provider | config.yaml `provider` | config.yaml `model` | Environment Variable |
|----------|----------------------|---------------------|---------------------|
| Groq (default) | `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| Google Gemini | `google` | `gemini-2.0-flash` | `GOOGLE_API_KEY` |
| OpenAI | `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `claude-3-haiku-20240307` | `ANTHROPIC_API_KEY` |
| Ollama (local) | `ollama` | `llama3.1` | No key needed |

Change `provider` and `model` in `config.yaml`, set the corresponding API key in `.env`, and restart the server.

---

## Widget Integration

To embed the chatbot into any existing web application or simulator dashboard:

```html
<script src="https://your-deployed-url.com/widget.js"></script>
```

This injects a floating chat bubble in the bottom-right corner. The widget uses Shadow DOM — its styles are fully encapsulated and will not interfere with the host page's CSS.

---

## Source Corpus

The chatbot was trained on (ingested) the following textbooks:

| Book | Pages |
|------|-------|
| Handbook of Solid Waste Management and Waste Minimization Technologies | 491 |
| Integrated Solid Waste Management: Engineering Principles (Tchobanoglous et al.) | 1,012 |
| Municipal Solid Waste Management in Developing Countries | 174 |
| Solid Waste Management Engineering | 322 |
| Solid Waste Management (New Edition) | 834 |
| Sustainable Solid Waste Management: A Systems Engineering Approach | 932 |
| **Total** | **3,765** |

---

## Cost Estimates

With Groq's free tier, the chatbot can handle approximately 1,000 queries/day at zero cost. For higher volumes:

| Provider | Estimated cost per 1,000 queries |
|----------|--------------------------------|
| Groq (Llama 3.3 70B) | $0.00 – $1.50 |
| Google Gemini 2.0 Flash | $0.50 – $2.00 |
| OpenAI GPT-4o-mini | $1.00 – $3.00 |
| Anthropic Claude 3.5 Haiku | $2.00 – $5.00 |

Embedding generation runs locally (sentence-transformers) — no API cost for document processing.

---

## License

MIT
