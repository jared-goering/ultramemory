"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode,
} from "react";
import type {
  MemoryNode,
  MemoryEdge,
  SearchResult,
  Stats,
  IngestItem,
} from "./types";

interface MemoryContextType {
  nodes: MemoryNode[];
  edges: MemoryEdge[];
  selectedNodeId: string | null;
  setSelectedNodeId: (id: string | null) => void;
  highlightedNodeIds: Set<string>;
  setHighlightedNodeIds: (ids: Set<string>) => void;
  searchResults: SearchResult[];
  setSearchResults: (results: SearchResult[]) => void;
  stats: Stats | null;
  entities: string[];
  newNodeIds: Set<string>;
  refreshGraph: () => Promise<void>;
  ingestFeed: IngestItem[];
  addToFeed: (items: IngestItem[]) => void;
}

const MemoryContext = createContext<MemoryContextType | null>(null);

export function useMemory() {
  const ctx = useContext(MemoryContext);
  if (!ctx) throw new Error("useMemory must be used within MemoryProvider");
  return ctx;
}

export function MemoryProvider({ children }: { children: ReactNode }) {
  const [nodes, setNodes] = useState<MemoryNode[]>([]);
  const [edges, setEdges] = useState<MemoryEdge[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [highlightedNodeIds, setHighlightedNodeIds] = useState<Set<string>>(
    new Set()
  );
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [entities, setEntities] = useState<string[]>([]);
  const [newNodeIds, setNewNodeIds] = useState<Set<string>>(new Set());
  const [ingestFeed, setIngestFeed] = useState<IngestItem[]>([]);
  const prevNodeIdsRef = useRef<Set<string>>(new Set());

  const refreshGraph = useCallback(async () => {
    try {
      const [graphRes, statsRes, entitiesRes] = await Promise.all([
        fetch("/api/graph"),
        fetch("/api/stats"),
        fetch("/api/entities"),
      ]);
      const graph = await graphRes.json();
      const statsData = await statsRes.json();
      const entitiesData = await entitiesRes.json();

      const prevIds = prevNodeIdsRef.current;
      const currentIds = new Set<string>(
        (graph.nodes as MemoryNode[]).map((n) => n.id)
      );
      const freshIds = new Set<string>();
      currentIds.forEach((id) => {
        if (!prevIds.has(id)) freshIds.add(id);
      });
      prevNodeIdsRef.current = currentIds;

      if (freshIds.size > 0 && prevIds.size > 0) {
        setNewNodeIds(freshIds);
        setTimeout(() => setNewNodeIds(new Set()), 2500);
      }

      setNodes(graph.nodes);
      setEdges(graph.edges);
      setStats({
        total: statsData.total_memories ?? statsData.total ?? 0,
        current: statsData.current_memories ?? statsData.current ?? 0,
        superseded: statsData.superseded_memories ?? statsData.superseded ?? 0,
        relations: statsData.relations ?? 0,
        categories: statsData.categories ?? {},
      });
      setEntities(
        Array.isArray(entitiesData) ? entitiesData : (entitiesData.entities ?? [])
      );
    } catch (e) {
      console.error("Failed to fetch graph data:", e);
    }
  }, []);

  useEffect(() => {
    refreshGraph();
  }, [refreshGraph]);

  const addToFeed = useCallback((items: IngestItem[]) => {
    setIngestFeed((prev) => [...items, ...prev].slice(0, 50));
  }, []);

  return (
    <MemoryContext.Provider
      value={{
        nodes,
        edges,
        selectedNodeId,
        setSelectedNodeId,
        highlightedNodeIds,
        setHighlightedNodeIds,
        searchResults,
        setSearchResults,
        stats,
        entities,
        newNodeIds,
        refreshGraph,
        ingestFeed,
        addToFeed,
      }}
    >
      {children}
    </MemoryContext.Provider>
  );
}
