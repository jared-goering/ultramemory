# ultramemory-claw

Automatic long-term memory for [OpenClaw](https://openclaw.ai) agents. No agent code changes needed.

## What it does

- **Before each agent turn:** Searches your memory DB for relevant context and injects it into the agent's prompt
- **After each agent response:** Extracts facts, decisions, and insights from the response and stores them as memories
- **On compaction:** Captures conversation content before it's summarized away
- **Tool:** Registers `memory_recall` for agents to do targeted deep searches

Powered by [ultramemory](https://github.com/jared-goering/ultramemory) (SQLite + local embeddings + LLM extraction).

## Install

```bash
# 1. Install the memory engine
pip install ultramemory[local]
ultramemory init
ultramemory serve  # starts API on :8642

# 2. Install the plugin
openclaw plugins install ultramemory-claw
openclaw gateway restart
```

## Config

In `openclaw.json` under `plugins.entries.ultramemory-claw.config`:

```json
{
  "enabled": true,
  "apiUrl": "http://127.0.0.1:8642",
  "topK": 5,
  "minSimilarity": 0.55,
  "ingestOnOutput": true,
  "ingestOnCompaction": true,
  "maxContextTokens": 2000,
  "excludeAgents": []
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `apiUrl` | `http://127.0.0.1:8642` | Supermemory API endpoint |
| `topK` | `5` | Max memories to inject per turn |
| `minSimilarity` | `0.55` | Minimum cosine similarity threshold |
| `ingestOnOutput` | `true` | Auto-extract memories from agent responses |
| `ingestOnCompaction` | `true` | Capture content before LCM compaction |
| `maxContextTokens` | `2000` | Token budget for injected memory context |
| `excludeAgents` | `[]` | Agent IDs to skip (no injection or extraction) |

## How it works

The plugin hooks into OpenClaw's lifecycle without any agent cooperation:

1. **`before_prompt_build`** - Extracts the latest user message, searches the memory DB, and prepends relevant memories as context
2. **`llm_output`** - Fires after the agent responds. Sends the response text to the ultramemory API for async fact extraction (fire-and-forget, won't slow down responses)
3. **`before_compaction`** - When LCM compacts old messages, captures the text before it's summarized

The memory engine uses LLM-based extraction (Haiku by default) to pull atomic facts with categories, confidence scores, and entity relationships. Search uses local embeddings (sentence-transformers) for fast semantic retrieval.

## Requirements

- OpenClaw >= 2026.0.0
- Python 3.10+
- `ultramemory[local]` package running with `ultramemory serve`

## License

MIT
