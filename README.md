<p align="center">
  <h1 align="center">Supermemory</h1>
  <p align="center">
    Local-first memory engine for AI agents.<br/>
    Atomic facts. Relational versioning. Temporal grounding. Zero cloud dependency.
  </p>
</p>

<p align="center">
  <a href="https://github.com/jared-goering/supermemory/actions"><img src="https://github.com/jared-goering/supermemory/actions/workflows/test.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/supermemory/"><img src="https://img.shields.io/pypi/v/supermemory.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/supermemory/"><img src="https://img.shields.io/pypi/pyversions/supermemory.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
</p>

<p align="center">
  <img src="docs/screenshot.jpg" alt="Supermemory 3D knowledge graph visualization" width="720">
</p>

---

Most AI memory solutions just append text to a vector store and call it a day. Supermemory takes a different approach: it extracts **atomic facts** from conversations, detects **relationships** between new and existing memories (update, extend, contradict), grounds everything in **time**, and stores it all in SQLite with local embeddings.

The result: your agent doesn't just remember *what* was said. It knows what changed, when, and why the old version was wrong.

## What makes it different

| Feature | Supermemory | Mem0 | Zep | LangMem |
|---------|:-----------:|:----:|:---:|:-------:|
| Relational versioning (update/contradict/extend) | ✅ | ❌ | ❌ | ❌ |
| Temporal grounding (event date vs. document date) | ✅ | ❌ | Partial | ❌ |
| Time-travel queries ("as of March 1st") | ✅ | ❌ | ❌ | ❌ |
| Local-first (SQLite + local embeddings) | ✅ | ❌ | ❌ | ❌ |
| Multi-agent shared memory | ✅ | ❌ | ✅ | ❌ |
| Entity profiles (auto-built) | ✅ | ✅ | ✅ | ❌ |
| No cloud account required | ✅ | ❌ | ❌ | ✅ |

**Relational versioning** means when you tell your agent "I moved from Seattle to Portland," it doesn't just add a new fact. It creates an UPDATE relation linking the new memory to the old one, marks the old memory as superseded, and preserves the full history. You can still query "where did I live in January?" and get the right answer.

**Temporal grounding** separates *when something was recorded* from *when it happened*. "Last Tuesday's meeting was cancelled" stores the event date as last Tuesday, not today. This makes time-based queries actually work.

## Quickstart

```bash
pip install supermemory[local]   # includes local embeddings (no API needed for search)

export ANTHROPIC_API_KEY=sk-ant-...   # or any litellm-supported provider

supermemory init
supermemory ingest --text "Alice started at Acme Corp in March. She moved from Seattle to Portland." --session demo --agent my-agent
supermemory search "Where does Alice work?"
```

Search uses local embeddings by default, so it's free and fast (~36ms warm). Ingestion requires an LLM for fact extraction (2-3 bounded calls per ingest).

## How it works

### Ingestion (2-3 LLM calls)

```
Text → Extract atomic facts → Detect relations to existing memories → Store + embed
```

1. **Extract** - LLM splits text into atomic facts, each with a category (decision, event, person, insight, etc.), confidence score, entity tags, and event date
2. **Relate** - New facts are compared to existing memories via embedding similarity. An LLM classifies the relationship: `updates`, `extends`, `contradicts`, `supports`, or `derives`
3. **Store** - Facts go into SQLite with their embedding (384-dim float32 BLOB). Superseded memories are marked but never deleted

### Search (zero LLM calls)

```
Query → Embed locally → Cosine similarity (matrix mul) → Filter by time/version → Return with relations
```

All search happens on-device. A single matrix-vector multiply scores every memory. Filtering by time window, version status, or entity is instant.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Ingestion Pipeline                      │
│                                                          │
│  Text/File ──► Extract (LLM) ──► Relate (LLM) ──► Store │
│                 atomic facts      UPDATE/EXTEND    SQLite │
│                 + categories      CONTRADICT/etc.  + BLOB │
│                 + entity tags                    embeddings│
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                 Search Pipeline (no LLM)                  │
│                                                          │
│  Query ──► Embed locally ──► Cosine sim ──► Time filter  │
│            sentence-        matrix mul     + version      │
│            transformers                    + expand rels  │
└──────────────────────────────────────────────────────────┘

Storage: SQLite (memories, relations, profiles) — single file, WAL mode
Embeddings: all-MiniLM-L6-v2 (384-dim, local, free)
LLM: Any provider via litellm (OpenAI, Anthropic, Ollama, etc.)
```

## CLI

```bash
supermemory init                                    # Create ~/.supermemory/ with config + empty DB
supermemory ingest --text "..." --session s --agent a  # Extract and store memories
supermemory ingest --file notes.md --session s --agent a  # Ingest from file
supermemory search "query"                          # Semantic search (current memories)
supermemory search "query" --all-versions           # Include superseded memories
supermemory search "query" --as-of 2025-06-01       # Time-travel query
supermemory history "Alice"                         # Version history for an entity
supermemory profile "Alice"                         # Auto-built entity profile
supermemory stats                                   # Database statistics
supermemory serve                                   # Start API server (default: localhost:8642)
```

## API

Start the server with `supermemory serve`, then:

### Ingest

```bash
curl -X POST http://localhost:8642/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice moved to Portland in March.",
    "session_key": "daily-standup",
    "agent_id": "kit",
    "document_date": "2025-03-15"
  }'
```

### Search

```bash
curl -X POST http://localhost:8642/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Where does Alice live?",
    "top_k": 10,
    "current_only": true
  }'
```

### All endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check (memory count, version) |
| `POST` | `/api/ingest` | Extract and store memories from text |
| `POST` | `/api/search` | Semantic search with filters |
| `POST` | `/api/recall` | Fast recall using cached embeddings |
| `POST` | `/api/startup-context` | Multi-query context for agent startup |
| `GET` | `/api/graph` | Full graph (nodes + edges) for visualization |
| `GET` | `/api/stats` | Database statistics by category |
| `GET` | `/api/history/{entity}` | Entity version history |
| `GET` | `/api/profile/{entity}` | Auto-built entity profile |
| `GET` | `/api/entities` | List all known entities |
| `POST` | `/api/cache/refresh` | Rebuild embedding cache |

## Multi-agent memory

Supermemory uses a single SQLite database with `agent_id` tagging. Every agent writes to the same store, and search spans all agents by default.

```python
# Agent A ingests a fact
supermemory ingest --text "Customer prefers email over phone" --agent sales-bot --session deal-42

# Agent B finds it later
supermemory search "How does the customer want to be contacted?" --agent support-bot
# → Returns the sales-bot's memory, with source attribution
```

This means your support agent knows what your sales agent learned, without explicit handoffs. Entity profiles aggregate knowledge across all agents automatically.

## Configuration

Config loads from (highest priority first):

1. Environment variables (`SUPERMEMORY_*`)
2. `./supermemory.yaml` (project-local)
3. `~/.supermemory/config.yaml` (user-global)
4. Built-in defaults

### Example config

```yaml
db_path: ~/.supermemory/memory.db

# LLM for extraction (any litellm-compatible model)
model: anthropic/claude-haiku-4-5

# Embeddings: "local" (free, on-device) or "litellm" (API-based)
embedding_provider: local
embedding_model: all-MiniLM-L6-v2
embedding_dim: 384

# API server
api_port: 8642
api_host: 0.0.0.0

# Dedup threshold (cosine similarity, 0.0-1.0)
dedup_threshold: 0.97

# Live ingest polling interval (seconds)
ingest_interval: 900

# Patterns to skip during ingestion
skip_patterns:
  - "HEARTBEAT_OK"

# Directories to scan for session files
session_scan_dirs:
  - ~/.openclaw/agents
```

### Environment variables

| Variable | Default |
|----------|---------|
| `SUPERMEMORY_DB_PATH` | `~/.supermemory/memory.db` |
| `SUPERMEMORY_MODEL` | `anthropic/claude-haiku-4-5` |
| `SUPERMEMORY_EMBEDDING_PROVIDER` | `local` |
| `SUPERMEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `SUPERMEMORY_EMBEDDING_DIM` | `384` |
| `SUPERMEMORY_API_PORT` | `8642` |
| `SUPERMEMORY_DEDUP_THRESHOLD` | `0.97` |

## Visualization

Supermemory includes a 3D knowledge graph UI built with React, Next.js, and [react-force-graph-3d](https://github.com/vasturiano/react-force-graph-3d). Nodes are colored by category, sized by connection count, and linked by relation type. Includes bloom post-processing for that glowing-brain look.

```bash
cd ui && pnpm install && pnpm dev
# Opens at http://localhost:3333
```

Features:
- 3D force-directed graph with bloom/glow effects
- Click any node to see the full memory, its relations, and version history
- Semantic search with real-time results
- Entity browser with auto-built profiles
- Live ingest panel for testing
- Stats dashboard with category breakdown

## Performance

Measured on Apple M4 (Mac mini, 16GB):

| Operation | Time |
|-----------|------|
| Search (warm, cached embeddings) | 36-94ms |
| Search (cold start, model load) | ~8s |
| Ingest (per text block) | 2-4s (LLM-bound) |
| Startup recall (multi-query) | <100ms |

Database tested with 9,000+ memories, 10,000+ relations, 1,000+ entity profiles. SQLite with WAL mode handles concurrent reads from multiple agents without issues.

## Development

```bash
git clone https://github.com/jared-goering/supermemory.git
cd supermemory
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
ruff format --check .
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. We welcome PRs for:

- New relation types
- Additional embedding providers
- Storage backends beyond SQLite
- Language support for fact extraction prompts
- UI improvements

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## License

[MIT](LICENSE)
