"use client";

import { useState, useEffect } from "react";
import { useMemory } from "../context";
import { getCategoryColor, EDGE_COLORS } from "../types";

export default function RightSidebar() {
  const {
    selectedNodeId,
    setSelectedNodeId,
    nodes,
    edges,
    highlightedNodeIds,
    setHighlightedNodeIds,
    searchResults,
    setSearchResults,
  } = useMemory();
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [entityHistory, setEntityHistory] = useState<
    Array<{ version?: number; content: string; date?: string }> | null
  >(null);
  const [entityProfile, setEntityProfile] = useState<{
    static_facts?: Record<string, unknown>;
    dynamic_facts?: Record<string, unknown>;
  } | null>(null);

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  const nodeRelations = edges.filter(
    (e) => e.source === selectedNodeId || e.target === selectedNodeId
  );

  const handleSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 10 }),
      });
      const data = await res.json();
      const results = data.results ?? data;
      setSearchResults(results);
      setHighlightedNodeIds(
        new Set(results.map((r: { id: string }) => r.id))
      );
    } catch (e) {
      console.error("Search failed:", e);
    }
    setSearching(false);
  };

  const clearSearch = () => {
    setSearchResults([]);
    setHighlightedNodeIds(new Set());
    setQuery("");
  };

  useEffect(() => {
    if (!selectedNode) {
      setEntityHistory(null);
      setEntityProfile(null);
      return;
    }
    // Extract a likely entity name from the content (first capitalized word or phrase)
    const words = selectedNode.content.split(/\s+/);
    const name =
      words.find((w) => /^[A-Z]/.test(w) && w.length > 1) ?? words[0];
    if (!name) return;

    Promise.allSettled([
      fetch(`/api/history/${encodeURIComponent(name)}`).then((r) => r.json()),
      fetch(`/api/profile/${encodeURIComponent(name)}`).then((r) => r.json()),
    ]).then(([historyResult, profileResult]) => {
      if (historyResult.status === "fulfilled") {
        const h = historyResult.value;
        setEntityHistory(h?.versions ?? (Array.isArray(h) ? h : null));
      } else {
        setEntityHistory(null);
      }
      if (profileResult.status === "fulfilled") {
        setEntityProfile(profileResult.value);
      } else {
        setEntityProfile(null);
      }
    });
  }, [selectedNode]);

  return (
    <div className="w-[360px] shrink-0 border-l border-zinc-800 flex flex-col bg-zinc-900/50 overflow-hidden">
      {/* Search */}
      <div className="p-4 border-b border-zinc-800">
        <h2 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-2">
          Search
        </h2>
        <div className="flex gap-2">
          <input
            className="flex-1 bg-zinc-800/80 rounded-lg px-3 py-2 text-sm border border-zinc-700/50 focus:border-blue-500/70 focus:outline-none placeholder:text-zinc-600 transition-colors"
            placeholder="Semantic search..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          />
          <button
            className="px-3 py-2 bg-zinc-800/80 hover:bg-zinc-700 rounded-lg text-sm transition-colors border border-zinc-700/50 disabled:opacity-40"
            onClick={handleSearch}
            disabled={searching}
          >
            {searching ? (
              <span className="w-3 h-3 border-2 border-zinc-400/30 border-t-zinc-400 rounded-full animate-spin inline-block" />
            ) : (
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="11" cy="11" r="8" />
                <path d="m21 21-4.3-4.3" />
              </svg>
            )}
          </button>
        </div>

        {/* Search Results */}
        {searchResults.length > 0 && (
          <div className="mt-3 space-y-2 max-h-64 overflow-y-auto">
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-zinc-500">
                {searchResults.length} results
              </span>
              <button
                className="text-[10px] text-zinc-500 hover:text-zinc-300 transition-colors"
                onClick={clearSearch}
              >
                Clear
              </button>
            </div>
            {searchResults.map((r) => (
              <button
                key={r.id}
                onClick={() => setSelectedNodeId(r.id)}
                className="w-full text-left bg-zinc-800/40 rounded-lg p-2.5 border border-zinc-700/30 hover:border-blue-500/30 transition-colors"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="flex items-center gap-1.5">
                    <span
                      className="w-1.5 h-1.5 rounded-full"
                      style={{ backgroundColor: getCategoryColor(r.category) }}
                    />
                    <span className="text-[10px] text-zinc-500 uppercase font-medium">
                      {r.category}
                    </span>
                  </span>
                  <span className="text-[10px] text-zinc-500 font-mono">
                    {(r.similarity * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-xs text-zinc-300 leading-relaxed">
                  {r.content}
                </p>
                <div className="mt-1.5 h-1 bg-zinc-700/50 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${r.similarity * 100}%`,
                      backgroundColor: getCategoryColor(r.category),
                      opacity: 0.7,
                    }}
                  />
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Detail Panel */}
      <div className="flex-1 overflow-y-auto p-4 min-h-0">
        {selectedNode ? (
          <>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest">
                Memory Detail
              </h2>
              <button
                onClick={() => setSelectedNodeId(null)}
                className="text-zinc-600 hover:text-zinc-300 transition-colors"
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M18 6 6 18M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="space-y-3">
              {/* Content */}
              <div className="bg-zinc-800/40 rounded-lg p-3 border border-zinc-700/30">
                <p className="text-sm text-zinc-200 leading-relaxed">
                  {selectedNode.content}
                </p>
              </div>

              {/* Metadata */}
              <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                <div>
                  <span className="text-zinc-500 text-[10px]">Category</span>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor: getCategoryColor(
                          selectedNode.category
                        ),
                      }}
                    />
                    <span className="text-zinc-300">
                      {selectedNode.category}
                    </span>
                  </div>
                </div>
                <div>
                  <span className="text-zinc-500 text-[10px]">Version</span>
                  <div className="font-mono mt-0.5 text-zinc-300">
                    v{selectedNode.version}
                  </div>
                </div>
                <div>
                  <span className="text-zinc-500 text-[10px]">Confidence</span>
                  <div className="mt-0.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1 bg-zinc-700/50 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500/70 rounded-full"
                          style={{
                            width: `${(selectedNode.confidence ?? 1) * 100}%`,
                          }}
                        />
                      </div>
                      <span className="font-mono text-zinc-400 text-[10px]">
                        {((selectedNode.confidence ?? 1) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
                <div>
                  <span className="text-zinc-500 text-[10px]">Status</span>
                  <div
                    className={`mt-0.5 font-medium ${
                      selectedNode.isCurrent
                        ? "text-green-400"
                        : "text-zinc-500"
                    }`}
                  >
                    {selectedNode.isCurrent ? "Current" : "Superseded"}
                  </div>
                </div>
              </div>

              {(selectedNode.documentDate || selectedNode.eventDate) && (
                <div className="text-xs space-y-0.5 text-zinc-400">
                  {selectedNode.documentDate && (
                    <div>
                      <span className="text-zinc-500">Document: </span>
                      <span className="font-mono">
                        {selectedNode.documentDate}
                      </span>
                    </div>
                  )}
                  {selectedNode.eventDate && (
                    <div>
                      <span className="text-zinc-500">Event: </span>
                      <span className="font-mono">
                        {selectedNode.eventDate}
                      </span>
                    </div>
                  )}
                </div>
              )}

              {selectedNode.session && (
                <div className="text-[10px] text-zinc-600 font-mono truncate">
                  Session: {selectedNode.session}
                </div>
              )}

              {/* Relations */}
              {nodeRelations.length > 0 && (
                <div>
                  <h3 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-2 mt-1">
                    Relations ({nodeRelations.length})
                  </h3>
                  <div className="space-y-1.5">
                    {nodeRelations.map((rel, i) => {
                      const edgeColor =
                        EDGE_COLORS[rel.type] ?? EDGE_COLORS.updates;
                      return (
                        <button
                          key={i}
                          onClick={() => {
                            const otherId =
                              rel.source === selectedNodeId
                                ? rel.target
                                : rel.source;
                            setSelectedNodeId(otherId);
                          }}
                          className="w-full text-left flex items-start gap-2 text-xs bg-zinc-800/30 hover:bg-zinc-800/60 rounded px-2.5 py-2 transition-colors"
                        >
                          <span
                            className="px-1.5 py-0.5 rounded text-[9px] font-mono uppercase shrink-0 mt-0.5"
                            style={{
                              backgroundColor: edgeColor + "20",
                              color: edgeColor,
                            }}
                          >
                            {rel.type}
                          </span>
                          <span className="text-zinc-400 leading-relaxed">
                            {rel.source === selectedNodeId
                              ? rel.targetContent
                              : rel.sourceContent}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Version History */}
              {entityHistory &&
                Array.isArray(entityHistory) &&
                entityHistory.length > 1 && (
                  <div>
                    <h3 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-2 mt-1">
                      Version History
                    </h3>
                    <div className="relative pl-4 border-l border-zinc-700/50 space-y-3">
                      {entityHistory.map((v, i) => (
                        <div key={i} className="relative">
                          <div className="absolute -left-[21px] w-2.5 h-2.5 rounded-full bg-zinc-800 border-2 border-zinc-600" />
                          <div className="text-xs">
                            <span className="font-mono text-zinc-500">
                              v{v.version ?? i + 1}
                            </span>
                            {v.date && (
                              <span className="text-zinc-600 ml-2 text-[10px]">
                                {v.date}
                              </span>
                            )}
                            <p className="text-zinc-400 mt-0.5 leading-relaxed">
                              {v.content}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

              {/* Entity Profile */}
              {entityProfile &&
                (entityProfile.static_facts || entityProfile.dynamic_facts) && (
                  <div>
                    <h3 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-2 mt-1">
                      Entity Profile
                    </h3>
                    <div className="bg-zinc-800/40 rounded-lg p-3 text-xs space-y-3 border border-zinc-700/30">
                      {entityProfile.static_facts &&
                        Object.keys(entityProfile.static_facts).length > 0 && (
                          <div>
                            <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
                              Static
                            </span>
                            <pre className="text-zinc-400 mt-1 whitespace-pre-wrap font-mono text-[10px] leading-relaxed">
                              {JSON.stringify(
                                entityProfile.static_facts,
                                null,
                                2
                              )}
                            </pre>
                          </div>
                        )}
                      {entityProfile.dynamic_facts &&
                        Object.keys(entityProfile.dynamic_facts).length > 0 && (
                          <div>
                            <span className="text-[10px] text-zinc-500 uppercase tracking-wider">
                              Dynamic
                            </span>
                            <pre className="text-zinc-400 mt-1 whitespace-pre-wrap font-mono text-[10px] leading-relaxed">
                              {JSON.stringify(
                                entityProfile.dynamic_facts,
                                null,
                                2
                              )}
                            </pre>
                          </div>
                        )}
                    </div>
                  </div>
                )}
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-zinc-600 gap-2">
            <svg
              width="32"
              height="32"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1"
              className="text-zinc-700"
            >
              <circle cx="12" cy="12" r="10" />
              <path d="M12 16v-4M12 8h.01" />
            </svg>
            <span className="text-sm">Select a node to view details</span>
          </div>
        )}
      </div>
    </div>
  );
}
