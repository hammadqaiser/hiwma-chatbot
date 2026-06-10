# Tests

160 automated tests organized by module. All tests use mocked dependencies — no API keys, no vector store, no network calls required.

## Running

```bash
# All tests
python -m pytest tests/ -v

# By module
python -m pytest tests/ingestion/      # 37 tests — PDF parsing, chunking, embedding, store
python -m pytest tests/retriever/      # 37 tests — vector, keyword, hybrid retrieval, formatting
python -m pytest tests/llm_gateway/    # 44 tests — all 5 providers, prompt templates, factory
python -m pytest tests/api/            # 42 tests — routes, orchestrator, models, chat history
```

## Structure

```
tests/
├── ingestion/
│   ├── test_parser.py          # PDF metadata extraction, chapter detection, edge cases
│   ├── test_chunker.py         # Overlap, min size, metadata propagation
│   ├── test_embedder.py        # Dimension checks, batch embedding
│   └── test_store.py           # Upsert, query, batching, collection management
├── retriever/
│   ├── test_vector_retriever.py    # Similarity search, filtering, ranking
│   ├── test_keyword_retriever.py   # BM25 tokenization, scoring, index rebuild
│   ├── test_hybrid_retriever.py    # Score combination, deduplication, weights
│   ├── test_context_formatter.py   # Context formatting, source deduplication
│   └── test_factory.py             # Config-driven retriever creation
├── llm_gateway/
│   ├── test_groq_llm.py        # Groq SDK mocking
│   ├── test_google_llm.py      # Google GenAI SDK mocking
│   ├── test_openai_llm.py      # OpenAI SDK mocking
│   ├── test_anthropic_llm.py   # Anthropic SDK mocking
│   ├── test_ollama_llm.py      # Ollama local mocking + health check
│   ├── test_prompt_templates.py # Prompt content validation
│   └── test_factory.py         # Provider creation, missing key errors
├── api/
│   ├── test_routes.py          # FastAPI TestClient endpoint tests
│   ├── test_orchestrator.py    # Pipeline flow, casual detection, mock retriever+LLM
│   ├── test_models.py          # Pydantic schema validation
│   └── test_chat_history.py    # SQLite session CRUD
├── e2e/                        # End-to-end tests (requires running services)
└── manual/                     # Manual verification scripts
```

## Conventions

- Test files mirror the source structure: `chatbot/ingestion/parser.py` → `tests/ingestion/test_parser.py`
- All external services (LLM APIs, ChromaDB, filesystem) are mocked
- Test classes are named `Test<Component>` and test functions `test_<behavior>`
