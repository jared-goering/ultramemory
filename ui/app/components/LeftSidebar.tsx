"use client";

import { useState } from "react";
import { useMemory } from "../context";
import { getCategoryColor } from "../types";

export default function LeftSidebar() {
  const { refreshGraph, stats, entities, ingestFeed, addToFeed, setSelectedNodeId } =
    useMemory();
  const [text, setText] = useState("");
  const [ingesting, setIngesting] = useState(false);

  const handleIngest = async () => {
    if (!text.trim()) return;
    setIngesting(true);
    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      const memories: Array<{ id?: string; content: string; category: string }> =
        data.memories ?? data.extracted ?? [];
      if (memories.length > 0) {
        addToFeed(
          memories.map((m) => ({
            id: m.id ?? crypto.randomUUID(),
            content: m.content,
            category: m.category,
            timestamp: new Date().toISOString(),
          }))
        );
      }
      setText("");
      await refreshGraph();
    } catch (e) {
      console.error("Ingest failed:", e);
    }
    setIngesting(false);
  };

  return (
    <div className="w-80 shrink-0 border-r border-zinc-800 flex flex-col bg-zinc-900/50 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800">
        <h1 className="text-lg font-bold tracking-tight flex items-center gap-2">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse" />
          Memory Engine
        </h1>
        <p className="text-[11px] text-zinc-500 mt-1">
          Visualize AI agent memory in motion
        </p>
      </div>

      {/* Ingestion */}
      <div className="p-4 border-b border-zinc-800">
        <h2 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-2">
          Ingest
        </h2>
        <textarea
          className="w-full bg-zinc-800/80 rounded-lg p-3 text-sm resize-none border border-zinc-700/50 focus:border-blue-500/70 focus:outline-none placeholder:text-zinc-600 transition-colors"
          rows={4}
          placeholder="Paste text to extract memories..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && e.metaKey) handleIngest();
          }}
        />
        <button
          className="mt-2 w-full bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-sm font-medium py-2 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          onClick={handleIngest}
          disabled={ingesting || !text.trim()}
        >
          {ingesting ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Extracting...
            </span>
          ) : (
            "Ingest"
          )}
        </button>
        <p className="text-[10px] text-zinc-600 mt-1.5 text-center">
          Cmd+Enter to submit
        </p>
      </div>

      {/* Feed */}
      <div className="flex-1 overflow-y-auto p-4 min-h-0">
        <h2 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-2">
          Recent Extractions
        </h2>
        {ingestFeed.length === 0 ? (
          <p className="text-xs text-zinc-600 italic">No memories extracted yet</p>
        ) : (
          <div className="space-y-2">
            {ingestFeed.map((item) => (
              <div
                key={item.id}
                className="bg-zinc-800/40 rounded-lg p-2.5 border border-zinc-700/30 hover:border-zinc-600/50 transition-colors cursor-default"
              >
                <div className="flex items-center gap-1.5 mb-1">
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ backgroundColor: getCategoryColor(item.category) }}
                  />
                  <span className="text-[10px] text-zinc-500 uppercase font-medium">
                    {item.category}
                  </span>
                </div>
                <p className="text-xs text-zinc-300 leading-relaxed">
                  {item.content}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Entities */}
      {entities.length > 0 && (
        <div className="p-4 border-t border-zinc-800 max-h-36 overflow-y-auto">
          <h2 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-2">
            Entities
          </h2>
          <div className="flex flex-wrap gap-1">
            {entities.map((name) => (
              <button
                key={name}
                onClick={() => setSelectedNodeId(name)}
                className="px-2 py-0.5 bg-zinc-800/60 hover:bg-zinc-700/60 rounded text-[11px] text-zinc-400 hover:text-zinc-200 border border-zinc-700/30 transition-colors"
              >
                {name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div className="p-4 border-t border-zinc-800 bg-zinc-900/80">
          <h2 className="text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-2">
            Stats
          </h2>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
            <div className="flex justify-between">
              <span className="text-zinc-500">Total</span>
              <span className="font-mono text-zinc-300">{stats.total}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500">Current</span>
              <span className="font-mono text-green-400">{stats.current}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500">Superseded</span>
              <span className="font-mono text-zinc-500">{stats.superseded}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500">Relations</span>
              <span className="font-mono text-zinc-300">{stats.relations}</span>
            </div>
          </div>
          {stats.categories && Object.keys(stats.categories).length > 0 && (
            <div className="flex gap-2 mt-2.5 flex-wrap">
              {Object.entries(stats.categories).map(([cat, count]) => (
                <div key={cat} className="flex items-center gap-1">
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: getCategoryColor(cat) }}
                  />
                  <span className="text-[10px] text-zinc-500">
                    {cat} <span className="font-mono">{count}</span>
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
