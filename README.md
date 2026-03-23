# OpenClaw Memory Engine

Local-first structured memory for AI agents. Extracts atomic facts from conversations, tracks versions and relations, and provides hybrid semantic search with temporal filtering.

## Features

- **Atomic memory extraction** — LLM splits conversations into single facts
- **Temporal versioning** — facts are versioned, old facts marked as superseded
- **Relation tracking** — memories linked via updates/extends/contradicts/supports/derives
- **Hybrid search** — embedding similarity + temporal filtering + relation expansion
- **Entity profiles** — auto-built from accumulated memories
- **Local-first** — SQLite + local embeddings, no cloud dependency for storage/retrieval
- **Provider-agnostic** — LLM calls via litellm (any provider)

## Setup

```bash
pip install -r requirements.txt
```

Set your LLM API key (e.g. `ANTHROPIC_API_KEY` for default model):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## CLI Usage

### Ingest text

```bash
python cli.py ingest --text "Jared lives at 123 Main St. He works at OpenClaw." --session "test" --agent "kit"
```

### Ingest from file

```bash
python cli.py ingest --file conversation.txt --session "test" --agent "kit"
```

### Search memories

```bash
python cli.py search "What is Jared's address?"
python cli.py search "Jared's job" --all-versions
python cli.py search "Jared's address" --as-of 2025-01-15
```

### View entity history

```bash
python cli.py history "Jared"
```

### View entity profile

```bash
python cli.py profile "Jared"
```

### Database stats

```bash
python cli.py stats
```

### Custom database path

```bash
python cli.py --db custom.db ingest --text "..." --session s1 --agent kit
```

## How It Works

### Ingestion (3 bounded LLM calls)

1. **Extract** — LLM extracts atomic facts from conversation text
2. **Relate** — New facts compared to existing memories for updates/extensions/contradictions
3. **Profile** — Entity profiles updated from accumulated memories

### Search (no LLM calls)

1. Embed query with sentence-transformers (all-MiniLM-L6-v2)
2. Cosine similarity against all memory embeddings
3. Filter by temporal/version constraints
4. Expand related memories
5. Attach source chunks for context

## Testing

```bash
python -m pytest test_engine.py -v
```

Tests mock LLM calls so no API key is needed.

## Architecture

```
Conversation → Extract (LLM) → Relate (LLM) → Profile (LLM) → SQLite
                                                                  ↑
Query → Embed → Cosine Similarity → Temporal Filter → Relations → Results
```

- **Embeddings**: all-MiniLM-L6-v2, 384-dim float32, stored as BLOB in SQLite
- **LLM**: litellm (default: anthropic/claude-haiku-4-5)
- **Storage**: SQLite with memories, memory_relations, and profiles tables
