export interface MemoryNode {
  id: string;
  content: string;
  category: string;
  confidence: number;
  documentDate: string | null;
  eventDate: string | null;
  isCurrent: boolean;
  version: number;
  session: string | null;
  agent: string | null;
  createdAt: string;
}

export interface MemoryEdge {
  source: string;
  target: string;
  type: "updates" | "contradicts" | "extends" | "supports" | "derives";
  context: string;
  sourceContent: string;
  targetContent: string;
}

export interface SearchResult {
  id: string;
  content: string;
  category: string;
  similarity: number;
  confidence: number;
  isCurrent: boolean;
}

export interface Stats {
  total: number;
  current: number;
  superseded: number;
  relations: number;
  categories: Record<string, number>;
}

export interface IngestItem {
  id: string;
  content: string;
  category: string;
  timestamp: string;
}

export const CATEGORY_COLORS: Record<string, string> = {
  person: "#3b82f6",
  project: "#22c55e",
  event: "#f59e0b",
  decision: "#a855f7",
  preference: "#ec4899",
};

export const CATEGORY_COLOR_DEFAULT = "#6b7280";

export function getCategoryColor(category: string): string {
  return CATEGORY_COLORS[category?.toLowerCase()] ?? CATEGORY_COLOR_DEFAULT;
}

export const EDGE_COLORS: Record<string, string> = {
  updates: "#a1a1aa",
  contradicts: "#ef4444",
  extends: "#3b82f6",
  supports: "#22c55e",
  derives: "#a855f7",
};
