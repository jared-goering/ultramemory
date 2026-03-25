/**
 * ultramemory-claw — Automatic agent memory plugin for OpenClaw
 *
 * Hooks into the agent lifecycle to:
 * 1. Inject relevant memories before each agent turn (before_prompt_build)
 * 2. Extract and store new memories from agent responses (llm_output)
 * 3. Capture conversation content before compaction (before_compaction)
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

type SupermemoryConfig = {
  enabled: boolean;
  apiUrl: string;
  topK: number;
  minSimilarity: number;
  excludeAgents: string[];
  ingestOnOutput: boolean;
  ingestOnCompaction: boolean;
  apiKey?: string;
  contextLabel: string;
  maxContextTokens: number;
};

const DEFAULT_CONFIG: SupermemoryConfig = {
  enabled: true,
  apiUrl: "http://127.0.0.1:8642",
  topK: 5,
  minSimilarity: 0.55,
  excludeAgents: [],
  ingestOnOutput: true,
  ingestOnCompaction: true,
  contextLabel: "Agent Memory",
  maxContextTokens: 2000,
};

function resolveConfig(raw: Record<string, unknown> | undefined): SupermemoryConfig {
  const r = raw ?? {};
  return {
    enabled: typeof r.enabled === "boolean" ? r.enabled : DEFAULT_CONFIG.enabled,
    apiUrl: typeof r.apiUrl === "string" && r.apiUrl.trim() ? r.apiUrl.trim() : DEFAULT_CONFIG.apiUrl,
    topK: typeof r.topK === "number" ? r.topK : DEFAULT_CONFIG.topK,
    minSimilarity:
      typeof r.minSimilarity === "number" ? r.minSimilarity : DEFAULT_CONFIG.minSimilarity,
    excludeAgents: Array.isArray(r.excludeAgents)
      ? r.excludeAgents.filter((x): x is string => typeof x === "string")
      : DEFAULT_CONFIG.excludeAgents,
    ingestOnOutput:
      typeof r.ingestOnOutput === "boolean" ? r.ingestOnOutput : DEFAULT_CONFIG.ingestOnOutput,
    ingestOnCompaction:
      typeof r.ingestOnCompaction === "boolean"
        ? r.ingestOnCompaction
        : DEFAULT_CONFIG.ingestOnCompaction,
    apiKey: typeof r.apiKey === "string" && r.apiKey.trim() ? r.apiKey.trim() : undefined,
    contextLabel:
      typeof r.contextLabel === "string" && r.contextLabel.trim()
        ? r.contextLabel.trim()
        : DEFAULT_CONFIG.contextLabel,
    maxContextTokens:
      typeof r.maxContextTokens === "number"
        ? r.maxContextTokens
        : DEFAULT_CONFIG.maxContextTokens,
  };
}

// ---------------------------------------------------------------------------
// HTTP helpers (lightweight, no external deps)
// ---------------------------------------------------------------------------

async function ultramemoryFetch(
  config: SupermemoryConfig,
  path: string,
  options: {
    method?: string;
    body?: unknown;
    timeoutMs?: number;
  } = {},
): Promise<unknown> {
  const url = `${config.apiUrl}${path}`;
  const method = options.method ?? "GET";
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (config.apiKey) {
    headers["X-API-Key"] = config.apiKey;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? 5000);

  try {
    const res = await fetch(url, {
      method,
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error(`Supermemory API ${method} ${path} returned ${res.status}`);
    }
    return await res.json();
  } finally {
    clearTimeout(timeout);
  }
}

// ---------------------------------------------------------------------------
// Agent exclusion matching (supports glob-style patterns)
// ---------------------------------------------------------------------------

function isAgentExcluded(agentId: string | undefined, patterns: string[]): boolean {
  if (!agentId || patterns.length === 0) return false;
  const id = agentId.trim().toLowerCase();
  return patterns.some((pattern) => {
    const p = pattern.trim().toLowerCase();
    if (p === id) return true;
    // Simple glob: "poly*" matches "polymarket"
    if (p.endsWith("*") && id.startsWith(p.slice(0, -1))) return true;
    return false;
  });
}

// ---------------------------------------------------------------------------
// Memory formatting
// ---------------------------------------------------------------------------

type MemoryResult = {
  id?: string;
  content?: string;
  text?: string; // fallback field name
  category?: string;
  confidence?: number;
  similarity?: number;
  agent_id?: string;
  source_session?: string;
  document_date?: string;
  created_at?: string;
};

function formatMemoriesAsContext(
  memories: MemoryResult[],
  config: SupermemoryConfig,
): string | undefined {
  if (!memories || memories.length === 0) return undefined;

  // Rough token budget: ~4 chars per token
  const charBudget = config.maxContextTokens * 4;
  const lines: string[] = [];
  let charCount = 0;

  for (const mem of memories) {
    const text = (mem.content ?? mem.text)?.trim();
    if (!text) continue;

    const similarity = mem.similarity != null ? ` [${(mem.similarity * 100).toFixed(0)}%]` : "";
    const category = mem.category ? ` (${mem.category})` : "";
    const line = `- ${text}${category}${similarity}`;

    if (charCount + line.length > charBudget) break;
    lines.push(line);
    charCount += line.length;
  }

  if (lines.length === 0) return undefined;

  return `## ${config.contextLabel}\n\nRelevant memories from prior sessions:\n\n${lines.join("\n")}`;
}

// ---------------------------------------------------------------------------
// Ingest helper
// ---------------------------------------------------------------------------

function fireAndForgetIngest(
  config: SupermemoryConfig,
  text: string,
  agentId: string | undefined,
  sessionKey: string | undefined,
  logger: { warn: (msg: string) => void },
): void {
  if (!text.trim()) return;

  ultramemoryFetch(config, "/api/ingest", {
    method: "POST",
    body: {
      text,
      agent_id: agentId ?? "unknown",
      session_key: sessionKey,
    },
    timeoutMs: 30000, // ingest can be slow (LLM extraction)
  }).catch((err: unknown) => {
    const message = err instanceof Error ? err.message : String(err);
    logger.warn(`[ultramemory] Ingest failed (non-blocking): ${message}`);
  });
}

// ---------------------------------------------------------------------------
// Plugin definition
// ---------------------------------------------------------------------------

const ultramemoryPlugin = {
  id: "ultramemory-claw",
  name: "Supermemory",
  description:
    "Automatic agent memory — injects relevant memories before each turn, extracts new memories from responses",

  configSchema: {
    parse(value: unknown) {
      const raw =
        value && typeof value === "object" && !Array.isArray(value)
          ? (value as Record<string, unknown>)
          : {};
      return resolveConfig(raw);
    },
  },

  register(api: OpenClawPluginApi) {
    const config = resolveConfig(
      api.pluginConfig && typeof api.pluginConfig === "object" && !Array.isArray(api.pluginConfig)
        ? (api.pluginConfig as Record<string, unknown>)
        : undefined,
    );

    if (!config.enabled) {
      api.logger.info("[ultramemory] Plugin disabled via config");
      return;
    }

    // Verify API is reachable on startup (non-blocking)
    ultramemoryFetch(config, "/api/health", { timeoutMs: 3000 })
      .then((data) => {
        const health = data as { status?: string; memories?: number; memory_count?: number; version?: string };
        const count = health.memories ?? health.memory_count;
        api.logger.info(
          `[ultramemory] Connected — ${count ?? "?"} memories, v${health.version ?? "?"}`,
        );
      })
      .catch(() => {
        api.logger.warn(
          `[ultramemory] API unreachable at ${config.apiUrl} — memories will not be available until the server starts`,
        );
      });

    // -----------------------------------------------------------------------
    // Hook 1: Inject relevant memories before each agent turn
    // -----------------------------------------------------------------------
    api.on("before_prompt_build", async (event, ctx) => {
      if (isAgentExcluded(ctx.agentId, config.excludeAgents)) return;

      const query = event.prompt?.trim();
      if (!query) return;

      try {
        const data = (await ultramemoryFetch(config, "/api/search", {
          method: "POST",
          body: {
            query,
            top_k: config.topK,
          },
          timeoutMs: 3000, // fast timeout — don't delay the response
        })) as { results?: MemoryResult[] };

        // Client-side similarity filtering
        const filtered = (data.results ?? []).filter(
          (r) => (r.similarity ?? 0) >= config.minSimilarity,
        );

        const context = formatMemoriesAsContext(filtered, config);
        if (context) {
          return { prependContext: context };
        }
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        // Silently degrade — agent works fine without memories
        api.logger.debug?.(`[ultramemory] Search failed: ${message}`);
      }
    });

    // -----------------------------------------------------------------------
    // Hook 2: Extract memories from agent output
    // -----------------------------------------------------------------------
    if (config.ingestOnOutput) {
      api.on("llm_output", (event, ctx) => {
        if (isAgentExcluded(ctx.agentId, config.excludeAgents)) return;

        const texts = event.assistantTexts ?? [];
        const combined = texts.join("\n\n").trim();
        if (!combined || combined.length < 20) return; // skip trivial responses

        // Build ingestable text: include both user prompt and assistant response
        // so the LLM extractor has full context
        const ingestText = `[User]: (context from agent turn)\n[Assistant]: ${combined}`;

        fireAndForgetIngest(config, ingestText, ctx.agentId, ctx.sessionKey, api.logger);
      });
    }

    // -----------------------------------------------------------------------
    // Hook 3: Capture content before compaction
    // -----------------------------------------------------------------------
    if (config.ingestOnCompaction) {
      api.on("before_compaction", async (event, ctx) => {
        if (isAgentExcluded(ctx.agentId, config.excludeAgents)) return;

        // If session file is available, we could read and ingest
        // But since we ingest on every llm_output, compaction content
        // should already be captured. This hook is a safety net for
        // conversations that started before the plugin was installed.
        const sessionFile = event.sessionFile;
        if (!sessionFile) return;

        try {
          const { readFileSync } = await import("node:fs");
          const raw = readFileSync(sessionFile, "utf8");
          const lines = raw.trim().split("\n");

          // Extract user+assistant messages from JSONL
          const messages: string[] = [];
          for (const line of lines) {
            try {
              const msg = JSON.parse(line) as {
                role?: string;
                content?: unknown;
              };
              if (msg.role === "user" || msg.role === "assistant") {
                const text =
                  typeof msg.content === "string"
                    ? msg.content
                    : Array.isArray(msg.content)
                      ? (msg.content as Array<{ type?: string; text?: string }>)
                          .filter((c) => c.type === "text" && c.text)
                          .map((c) => c.text)
                          .join("\n")
                      : "";
                if (text.trim()) {
                  messages.push(`[${msg.role}]: ${text.trim()}`);
                }
              }
            } catch {
              // skip malformed lines
            }
          }

          if (messages.length > 0) {
            const combined = messages.join("\n\n");
            fireAndForgetIngest(config, combined, ctx.agentId, ctx.sessionKey, api.logger);
          }
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : String(err);
          api.logger.debug?.(`[ultramemory] Compaction ingest failed: ${message}`);
        }
      });
    }

    // -----------------------------------------------------------------------
    // Register a search tool so agents can also query memories explicitly
    // -----------------------------------------------------------------------
    api.registerTool((ctx) => ({
      name: "memory_recall",
      label: "Memory Recall",
      description:
        "Search your long-term memory for relevant facts, decisions, and context from prior sessions. Memories are also injected automatically, but use this for targeted deep recall.",
      parameters: {
        type: "object" as const,
        properties: {
          query: {
            type: "string",
            description: "Natural language search query",
          },
          top_k: {
            type: "number",
            description: "Number of results to return (default: 10, max: 50)",
          },
          entity: {
            type: "string",
            description: "Filter by entity name (e.g., a person, project, or concept)",
          },
        },
        required: ["query"] as const,
      },
      async execute(_toolCallId: string, params: Record<string, unknown>) {
        const query = typeof params.query === "string" ? params.query.trim() : "";
        if (!query) {
          return {
            content: [{ type: "text" as const, text: "Error: query is required" }],
            details: { error: "Query is required" },
          };
        }

        const topK = typeof params.top_k === "number" ? Math.min(params.top_k, 50) : 10;
        const entity = typeof params.entity === "string" ? params.entity.trim() : undefined;

        try {
          const data = (await ultramemoryFetch(config, "/api/search", {
            method: "POST",
            body: {
              query,
              top_k: topK,
            },
            timeoutMs: 5000,
          })) as { results?: MemoryResult[] };

          const results = (data.results ?? []).map((r) => ({
            text: r.content ?? r.text,
            category: r.category,
            confidence: r.confidence,
            similarity: r.similarity != null ? Math.round(r.similarity * 100) / 100 : undefined,
            agent: r.agent_id,
            date: r.document_date ?? r.created_at,
          }));

          const resultText = results.length > 0
            ? results.map((r) => `- ${r.text} (${r.category ?? "general"}, ${r.similarity ? r.similarity * 100 + "%" : "?"} match)`).join("\n")
            : "No relevant memories found.";

          return {
            content: [{ type: "text" as const, text: `Found ${results.length} memories:\n\n${resultText}` }],
            details: { count: results.length, results },
          };
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : String(err);
          return {
            content: [{ type: "text" as const, text: `Memory search failed: ${message}` }],
            details: { error: message },
          };
        }
      },
    }));

    api.logger.info(
      `[ultramemory] Plugin registered — topK=${config.topK}, ingestOnOutput=${config.ingestOnOutput}, excludeAgents=[${config.excludeAgents.join(",")}]`,
    );
  },
};

export default ultramemoryPlugin;
