# Ultramemory v0.3 Build Plan

## Goal
Move from 55% → 75%+ on LongMemEval by implementing structured event retrieval and hybrid query routing.

## Current Failure Analysis (from GPT 5.4)
- **6/10 failures**: Retrieval (right facts not in top-K)
- **3/10 failures**: Reasoning (temporal math wrong despite having facts)
- **1/10 failures**: Aggregation (counting/combining across sessions)
- **4 secondary**: Extraction (facts poorly formed or missing)

## Architecture Changes

### Sprint 1: Structured Event Layer (2-3 days)
The single highest-impact change. Adds typed event records alongside free-text memories.

**New table: `events`**
```sql
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    memory_id TEXT REFERENCES memories(id),  -- link to source memory
    agent_id TEXT NOT NULL,

    -- Event typing
    event_type TEXT NOT NULL,        -- 'wedding_attendance', 'purchase', 'trip', 'certification', 'exercise', etc.
    event_category TEXT,             -- broad category: 'social', 'education', 'health', 'travel', 'work', 'purchase'

    -- Temporal
    event_date TEXT,                 -- ISO date (resolved, not relative)
    event_end_date TEXT,             -- for multi-day events
    duration_minutes INTEGER,        -- if known

    -- Entity/detail
    description TEXT NOT NULL,       -- "attended cousin Rachel's wedding"
    participants TEXT,               -- JSON array of names
    location TEXT,
    status TEXT DEFAULT 'completed', -- 'planned', 'completed', 'cancelled', 'habitual'

    -- Dedup
    canonical_event_id TEXT,         -- groups duplicate mentions of same event

    -- Meta
    created_at TEXT DEFAULT (datetime('now')),
    source_session TEXT
);

CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_date ON events(event_date);
CREATE INDEX idx_events_agent ON events(agent_id);
CREATE INDEX idx_events_category ON events(event_category);
CREATE INDEX idx_events_canonical ON events(canonical_event_id);
```

**Extraction changes in `engine.py`:**
- After extracting atomic facts, run a second LLM pass on the same chunk
- Prompt: "Extract any events (things that happened, were planned, or are habitual) with: type, date, duration, participants, location, status"
- Date resolution: LLM gets the session date as reference, must output ISO dates (not "last Tuesday")
- Store in events table, linked to source memory via memory_id

**New extraction prompt (addition to existing):**
```
In addition to facts, extract EVENTS from this conversation. An event is something
that happened, is planned, or is habitual. For each event, provide:
- event_type: specific label (wedding_attendance, book_finished, trip, exercise_session,
  purchase, certification_completed, networking_event, religious_service, etc.)
- event_category: broad category (social, education, health, travel, work, purchase, entertainment)
- event_date: resolved ISO date (use session date {session_date} as reference for "yesterday", "last week", etc.)
- event_end_date: if multi-day
- duration_minutes: if mentioned
- description: one-line summary
- participants: list of names involved
- location: if mentioned
- status: completed | planned | cancelled | habitual
```

**Estimated work:**
- Schema + migration: 2 hours
- Extraction prompt + LLM integration: 4 hours
- Event dedup (canonical_event_id assignment): 4 hours
- Tests: 3 hours

### Sprint 2: Query Router + Hybrid Search (2 days)
Classify incoming queries and route to the right retrieval strategy.

**Query classification (lightweight LLM call or rule-based):**
```python
class QueryType(Enum):
    FACT_LOOKUP = "fact_lookup"           # "What's my dog's name?"
    TEMPORAL_LOOKUP = "temporal_lookup"    # "What did I buy 10 days ago?"
    EVENT_COUNTING = "event_counting"     # "How many weddings this year?"
    AGGREGATION = "aggregation"           # "Total days in Japan + Chicago?"
    PREFERENCE = "preference"             # "What kind of food do I like?"
    KNOWLEDGE_STATE = "knowledge_state"   # "How many followers do I have?"
```

**Routing logic:**
```python
def hybrid_search(query, agent_id, reference_date):
    query_type = classify_query(query)  # fast LLM or rules

    if query_type == QueryType.FACT_LOOKUP:
        # Standard embedding search (current behavior)
        return embedding_search(query, agent_id, top_k=10)

    elif query_type == QueryType.TEMPORAL_LOOKUP:
        # Parse temporal expression, resolve to date, search events
        target_date = resolve_temporal(query, reference_date)
        events = search_events_by_date(target_date, agent_id, window_days=2)
        memories = embedding_search(query, agent_id, top_k=5)
        return merge_results(events, memories)

    elif query_type == QueryType.EVENT_COUNTING:
        # Extract event type + time window, query events table
        event_type, time_window = parse_counting_query(query)
        events = search_events_by_type(event_type, time_window, agent_id)
        # Deduplicate by canonical_event_id
        unique_events = deduplicate_events(events)
        return unique_events

    elif query_type == QueryType.AGGREGATION:
        # Multi-entity: extract entities, search each, combine
        entities = extract_query_entities(query)
        all_results = []
        for entity in entities:
            results = embedding_search(entity, agent_id, top_k=10)
            events = search_events_by_entity(entity, agent_id)
            all_results.extend(merge_results(events, results))
        return all_results

    else:
        # Fallback to current behavior
        return embedding_search(query, agent_id, top_k=10)
```

**Estimated work:**
- Query classifier: 4 hours (start rule-based, upgrade to LLM later)
- Temporal expression parser: 4 hours (dateutil + custom patterns)
- Event table search functions: 3 hours
- Result merging + dedup: 3 hours
- Integration into server.py search endpoint: 2 hours
- Tests: 3 hours

### Sprint 3: Deterministic Temporal Reasoning (1 day)
Remove date math from the answer LLM entirely.

**Components:**
1. **Reference date resolver**: Extract from query context or use latest session timestamp
2. **Temporal expression parser**: "10 days ago" → date, "last month" → date range, "this year" → date range
3. **Date arithmetic engine**: Compute differences, durations, ordering
4. **Pre-computed temporal context**: Before calling the answer LLM, inject computed values

**Example flow:**
```
Query: "How many days between Sunday mass and Ash Wednesday service?"
→ Parse: two event references
→ Search events: find mass (event_date: Feb 5) and service (event_date: Mar 8)
→ Compute: Mar 8 - Feb 5 = 31 days
→ Inject into answer prompt: "The time between these events is 31 days."
→ LLM just formats the answer
```

**Estimated work:**
- Temporal parser: 3 hours
- Date arithmetic module: 2 hours
- Answer prompt injection: 2 hours
- Tests: 2 hours

### Sprint 4: Benchmark + Polish (1 day)
- Re-run full 20-question benchmark
- Compare results per-category against baseline (55%)
- Fix any regressions in single-session categories
- Update README with new numbers
- Tag v0.3.0 and push to PyPI

## Expected Impact (per failure)

| Question Type | Current | Fix | Expected |
|---|---|---|---|
| knowledge-update (2) | 2/2 | No change | 2/2 |
| single-session-assistant (2) | 2/2 | No change | 2/2 |
| single-session-preference (2) | 2/2 | No change | 2/2 |
| single-session-user (2) | 1/2 | Event layer catches certification | 2/2 |
| multi-session (4) | 0/4 | Event counting + dedup | 3/4 |
| temporal-reasoning (8) | 3/8 | Event layer + deterministic math | 6/8 |
| **Total** | **10/20 (50%)** | | **17/20 (85%)** |

Conservative estimate: 70%. Optimistic: 85%.

## Files to Create/Modify

**New files:**
- `ultramemory/events.py` — Event extraction, storage, search, dedup
- `ultramemory/query_router.py` — Query classification and routing
- `ultramemory/temporal.py` — Temporal expression parsing + date arithmetic
- `tests/test_events.py`
- `tests/test_query_router.py`
- `tests/test_temporal.py`

**Modified files:**
- `ultramemory/engine.py` — Add event extraction to ingest pipeline, schema migration
- `ultramemory/server.py` — Add hybrid search endpoint, event search endpoint
- `ultramemory/models.py` — Add Event model, QueryType enum

## Non-Goals for v0.3
- Recall expansion / multi-query retrieval (v0.4)
- Cross-encoder reranking (v0.4)
- BM25 sparse retrieval (v0.4, need to evaluate if worth the dependency)
- Fine-tuned extraction model (v0.5+)

## Cost Estimate
- Event extraction adds ~1 LLM call per ingested chunk (Haiku/Flash tier, ~$0.005/chunk)
- Query routing adds ~1 lightweight LLM call per search (~$0.001/query)
- Total per-search cost increase: negligible for rule-based router, ~$0.001 for LLM router

## Timeline
- Sprint 1 (Event Layer): Days 1-3
- Sprint 2 (Query Router): Days 3-5
- Sprint 3 (Temporal): Day 5-6
- Sprint 4 (Benchmark): Day 6-7
- **Total: ~7 days to v0.3.0**
