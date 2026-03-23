# OpenClaw Memory Engine — Design Doc
*Inspired by Supermemory's SOTA architecture, adapted for local-first agent memory*

## Problem Statement

OpenClaw agents currently use flat markdown files (MEMORY.md + memory/*.md) with basic semantic search (memory_search). This works but has critical gaps:

1. **No version history** — facts are overwritten, not versioned ("address changed" loses the old address)
2. **No temporal reasoning** — can't answer "what was true at time X?" or "what changed between dates?"
3. **No relationship tracking** — memories are isolated text blobs, not linked facts
4. **Stale data accumulates silently** — no mechanism to detect outdated facts
5. **Search is chunk-level** — returns document sections, not atomic facts
6. **Multi-agent memory is fragmented** — each agent's context is siloed

## Architecture

### Core Data Model (SQLite, local-first)

```sql
-- Atomic memories: single facts extracted from conversations
CREATE TABLE memories (
    id TEXT PRIMARY KEY,              -- uuid
    content TEXT NOT NULL,            -- atomic fact: "Jared lives at 742 Evergreen Terrace, Springfield, IL"
    category TEXT,                    -- person, preference, project, decision, event, insight
    confidence REAL DEFAULT 1.0,      -- 0-1, decays over time
    
    -- Temporal grounding (Supermemory's key insight)
    document_date TEXT NOT NULL,      -- when the conversation happened (ISO)
    event_date TEXT,                  -- when the fact/event actually occurred (ISO, nullable)
    
    -- Source tracking
    source_session TEXT,              -- session key that generated this memory
    source_agent TEXT,                -- which agent extracted it
    source_chunk TEXT,                -- original conversation chunk (for context retrieval)
    
    -- Versioning
    version INTEGER DEFAULT 1,
    is_current BOOLEAN DEFAULT TRUE,  -- FALSE = superseded by a newer memory
    superseded_by TEXT,               -- id of the memory that replaced this one
    
    -- Embedding for semantic search
    embedding BLOB,                   -- local embedding vector (float32)
    
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Relationships between memories (knowledge chains)
CREATE TABLE memory_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_memory TEXT NOT NULL,
    to_memory TEXT NOT NULL,
    relation TEXT NOT NULL,           -- 'updates' | 'extends' | 'derives' | 'contradicts' | 'supports'
    context TEXT,                     -- why this relationship exists
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (from_memory) REFERENCES memories(id),
    FOREIGN KEY (to_memory) REFERENCES memories(id)
);

-- User/entity profiles built from memories
CREATE TABLE profiles (
    id TEXT PRIMARY KEY,
    entity_name TEXT NOT NULL,        -- "Jared", "Acme Corp", etc.
    static_profile TEXT,              -- core stable facts (JSON)
    dynamic_profile TEXT,             -- recent/evolving facts (JSON)
    updated_at TEXT DEFAULT (datetime('now'))
);
```

### Ingestion Pipeline

```
Conversation ends
       ↓
┌─────────────────────────────────────────────────┐
│ 1. CHUNK: Split session into semantic blocks     │
│    (natural turn boundaries, topic shifts)        │
└──────────────────────┬──────────────────────────┘
                       ↓
┌──────────────────────┴──────────────────────────┐
│ 2. EXTRACT: LLM extracts atomic memories         │
│    Prompt: "Extract single facts from this chunk. │
│    For each fact, identify:                       │
│    - The atomic fact (one sentence)               │
│    - Category (person/preference/project/etc)     │
│    - Event date (when did this happen/will happen) │
│    - Confidence (how certain is this fact?)"       │
└──────────────────────┬──────────────────────────┘
                       ↓
┌──────────────────────┴──────────────────────────┐
│ 3. RELATE: For each new memory, search existing   │
│    memories for conflicts/extensions               │
│    Prompt: "Does this new fact UPDATE, EXTEND,     │
│    or CONTRADICT any existing memory?"             │
│    - updates → mark old as superseded              │
│    - extends → link as extension                   │
│    - contradicts → flag for human review           │
│    - derives → link as inference                   │
└──────────────────────┬──────────────────────────┘
                       ↓
┌──────────────────────┴──────────────────────────┐
│ 4. EMBED: Generate local embedding vector         │
│    (sentence-transformers, runs on Apple Silicon)  │
└──────────────────────┬──────────────────────────┘
                       ↓
┌──────────────────────┴──────────────────────────┐
│ 5. STORE: Insert memory + relations + chunk ref   │
│    Update profiles if entity-related               │
└─────────────────────────────────────────────────┘
```

### Retrieval Pipeline (replaces memory_search)

```
Query: "What's Jared's address?"
       ↓
┌─────────────────────────────────────────────────┐
│ 1. SEMANTIC SEARCH on atomic memories             │
│    (not raw chunks — high signal, low noise)      │
│    → Top K memory matches                         │
└──────────────────────┬──────────────────────────┘
                       ↓
┌──────────────────────┴──────────────────────────┐
│ 2. TEMPORAL FILTER                                │
│    - Only return is_current=TRUE (latest version)  │
│    - Include version chain for context             │
│    - Respect event_date for "as of" queries        │
└──────────────────────┬──────────────────────────┘
                       ↓
┌──────────────────────┴──────────────────────────┐
│ 3. EXPAND: Follow relations                       │
│    - If memory has 'extends' → include extensions  │
│    - If memory was 'updated' → include history     │
│    - Pull related profile data                     │
└──────────────────────┬──────────────────────────┘
                       ↓
┌──────────────────────┴──────────────────────────┐
│ 4. INJECT SOURCE CHUNKS                           │
│    For each memory hit, attach the original chunk  │
│    (preserves nuance lost in atomic extraction)    │
└──────────────────────┬──────────────────────────┘
                       ↓
│ Return: memories + chunks + relations + profiles  │
```

## Implementation Plan

### Phase 1: Core Engine (Weekend Build)
- [ ] SQLite schema + Python module (`memory_engine.py`)
- [ ] Ingestion: extract atomic memories from conversation text via LLM
- [ ] Relational versioning: updates/extends/derives detection
- [ ] Temporal grounding: dual timestamps
- [ ] Local embeddings: sentence-transformers on MPS
- [ ] Hybrid search: semantic on memories → inject source chunks
- [ ] CLI: `memory ingest <session>`, `memory search <query>`, `memory history <entity>`
- **Deliverable:** Working local memory engine, tested on our existing conversation logs

### Phase 2: OpenClaw Integration
- [ ] OpenClaw skill wrapping the engine
- [ ] Hook into session end → auto-ingest conversation
- [ ] Replace memory_search tool with engine's hybrid search
- [ ] Multi-agent support: all agents read/write to shared memory DB
- [ ] Profile auto-building from accumulated memories
- [ ] Memory decay: confidence drops over time unless reinforced
- **Deliverable:** Drop-in OpenClaw skill that improves any agent's memory

### Phase 3: Open Source Package
- [ ] Clean API: `pip install openclaw-memory`
- [ ] Provider-agnostic LLM calls (works with any model)
- [ ] Pluggable embedding backends (sentence-transformers, OpenAI, Ollama)
- [ ] Export/import (migrate from MEMORY.md to structured memory)
- [ ] LongMemEval benchmark runner (prove our numbers)
- [ ] Docs + examples
- **Deliverable:** Published package + OpenClaw skill on ClawHub

## Key Design Decisions

### Local-first, no SaaS dependency
Everything runs on SQLite + local embeddings. No API calls needed for storage/retrieval. LLM calls only during ingestion (extracting memories + detecting relations). This means:
- Works offline
- No data leaves the machine
- Fast retrieval (SQLite + vector similarity)
- Agent can use cheap/fast model for extraction, expensive model for reasoning

### Backward compatible with MEMORY.md
During Phase 2, both systems run in parallel:
- MEMORY.md continues as the human-readable curated layer
- Structured memory DB handles machine retrieval
- Optional: auto-generate MEMORY.md from structured memories

### LLM calls are bounded
Supermemory caps at 3 LLM calls per ingestion. We should too:
1. Extract atomic memories from chunk
2. Check relations against existing memories  
3. (Optional) Update profiles

This keeps cost manageable even with high conversation volume.

### Embedding model
- **sentence-transformers/all-MiniLM-L6-v2** — 384-dim, runs fast on MPS, good enough for memory search
- ~80MB model, loads once, stays in memory
- Alternative: nomic-embed-text via Ollama (768-dim, slightly better quality)

## What This Doesn't Do (Yet)
- **Proactive recall** — "you mentioned X 3 weeks ago, is that still relevant?"
- **Memory consolidation** — merging related memories over time (like sleep does for humans)
- **Cross-agent reasoning** — "Poly noticed X, does that affect what Kit knows about Y?"
- **Forgetting** — intentionally dropping low-confidence, unreinforced memories

These are Phase 4+ features.

## Rough Token Cost Estimate
- Ingestion: ~1K tokens per conversation chunk × 3 LLM calls = ~3K tokens per chunk
- Average session: ~5-10 chunks = 15-30K tokens per session ingestion
- At Claude Sonnet rates: ~$0.01-0.02 per session ingestion
- Retrieval: no LLM cost (pure embedding similarity + SQLite)
- Monthly estimate (10 sessions/day): ~$3-6/month

## Competition / Prior Art
| System | Approach | Limitation |
|--------|----------|------------|
| Supermemory | Cloud SaaS, relational versioning | Proprietary, requires API |
| Mem0 | Key-value memory store | No temporal reasoning, no relations |
| Zep | Temporal knowledge graph | Complex setup, cloud-first |
| Letta (MemGPT) | Self-editing memory blocks | Limited to in-context memory |
| OpenClaw (current) | Flat files + semantic search | No versioning, no relations |
| **This project** | Local SQLite + atomic memories + relations | Best of all: local, versioned, temporal |
