# Backend Modules

The `chatbot/` package contains four backend modules and a frontend directory. Each module is independently testable and replaceable — they communicate through abstract interfaces (`BaseRetriever`, `BaseLLM`) and Pydantic schemas.

## Module Layout

```
chatbot/
├── config.py              # YAML + environment config loader
├── ingestion/             # M1: Document processing pipeline
├── retriever/             # M2: Search and retrieval engine
├── llm_gateway/           # M3: LLM provider abstraction
├── api/                   # M4: FastAPI application and orchestration
└── frontend/              # M5: Standalone chat UI and embeddable widget
```

---

## M1: Ingestion (`ingestion/`)

Converts raw PDF files into an indexed, searchable vector store.

| File | Responsibility |
|------|---------------|
| `parser.py` | Extracts text page-by-page using PyMuPDF. Captures metadata: book title, detected chapter headings, and page numbers. |
| `chunker.py` | Splits page text into overlapping chunks (default: 1000 tokens, 200 overlap). Respects paragraph boundaries. Each chunk inherits its source metadata. |
| `embedder.py` | Generates 384-dimensional vectors using `all-MiniLM-L6-v2` (runs locally). Handles the deprecated `get_sentence_embedding_dimension` API gracefully. |
| `store.py` | Manages ChromaDB. Upserts in batches of 5,000 to stay within ChromaDB's per-call limit of 5,461 items. Supports query, count, and collection listing. |
| `cli.py` | CLI entry point: `python -m chatbot.ingestion.cli --source ./Books`. Orchestrates the full parse → chunk → embed → store flow with progress output. |

---

## M2: Retriever (`retriever/`)

Accepts a natural-language query and returns the most relevant document chunks.

| File | Responsibility |
|------|---------------|
| `base.py` | Defines `BaseRetriever` (abstract) and `RetrievalResult` (dataclass with text, score, book_title, chapter, page_number, page_range, chunk_id). |
| `vector_retriever.py` | Wraps ChromaDB similarity search. Embeds the query at runtime, retrieves top-k by cosine distance. |
| `keyword_retriever.py` | BM25 lexical search using `rank-bm25`. Catches exact technical terms that semantic search can miss. Tokenizes with punctuation removal and stopword filtering. |
| `hybrid_retriever.py` | Combines vector (70%) and keyword (30%) results. Normalizes scores to [0, 1], deduplicates by chunk ID, and returns a merged ranked list. |
| `context_formatter.py` | Formats retrieved chunks into a structured text block for LLM consumption. Extracts and deduplicates source metadata for the citation UI. |
| `factory.py` | Config-driven factory: `create_retriever(config)` returns the appropriate retriever instance based on `config.yaml → retriever.type`. |

---

## M3: LLM Gateway (`llm_gateway/`)

Abstracts text generation behind a common interface. Switching providers is a config change, not a code change.

| File | Responsibility |
|------|---------------|
| `base.py` | Defines `BaseLLM` (abstract), `LLMResponse`, and `TokenUsage`. All providers implement `generate(query, context) → LLMResponse`. |
| `prompt_templates.py` | System prompt (anti-hallucination, no inline citations, friendly personality), user prompt template, and casual conversation prompt. |
| `groq_llm.py` | Groq SDK integration. Default provider. |
| `google_llm.py` | Google GenAI SDK integration. Uses `contents` parameter with separate system instruction. |
| `openai_llm.py` | OpenAI SDK integration. Also supports Azure OpenAI via `base_url`. |
| `anthropic_llm.py` | Anthropic SDK integration. Uses the separate `system` parameter (not a message role). |
| `ollama_llm.py` | Local Ollama integration via OpenAI-compatible API. Includes health check. No API key required. |
| `factory.py` | Config-driven factory: `create_llm(config)` validates the API key exists and returns the appropriate provider instance. |

---

## M4: API (`api/`)

FastAPI application that wires the pipeline together and serves the frontend.

| File | Responsibility |
|------|---------------|
| `main.py` | Application factory. Configures CORS, registers routes, mounts static files from `frontend/`, and manages the application lifespan (startup/shutdown of shared instances). |
| `models.py` | Pydantic schemas: `ChatRequest`, `ChatResponse`, `SourceCitation`, `HealthResponse`, `DocumentInfo`. |
| `chat_history.py` | Async SQLite storage for chat sessions. Stores messages with timestamps and session IDs. Uses `aiosqlite` for non-blocking I/O. |
| `routes/chat.py` | `POST /api/chat` — accepts a message and optional session ID, runs the orchestrator, persists the exchange, returns the response. |
| `routes/health.py` | `GET /api/health` — returns service status, active LLM provider and model, retriever type. |
| `routes/documents.py` | `GET /api/documents` — lists ingested documents with chunk counts. |
| `services/orchestrator.py` | Core business logic. Detects casual messages (regex patterns → skip RAG), or runs the full retrieve → format → generate pipeline. Created once at startup, reused across requests. |
