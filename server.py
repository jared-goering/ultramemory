"""FastAPI wrapper around MemoryEngine for the visualization UI."""

import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from memory_engine import MemoryEngine

app = FastAPI(title="Memory Engine API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.environ.get("MEMORY_DB", "memory.db")
engine = MemoryEngine(db_path=DB_PATH)


class IngestRequest(BaseModel):
    text: str
    session_key: str = "ui"
    agent_id: str = "user"
    document_date: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    current_only: bool = True
    as_of_date: Optional[str] = None


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    memories = engine.ingest(
        req.text,
        session_key=req.session_key,
        agent_id=req.agent_id,
        document_date=req.document_date,
    )
    return {"memories": memories, "count": len(memories)}


@app.post("/api/search")
async def search(req: SearchRequest):
    results = engine.search(
        req.query,
        top_k=req.top_k,
        current_only=req.current_only,
        as_of_date=req.as_of_date,
    )
    return {"results": results, "count": len(results)}


@app.get("/api/graph")
async def graph():
    """Return all memories and relations as nodes + edges for visualization."""
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get all memories
    rows = conn.execute(
        """SELECT id, content, category, confidence,
                  document_date, event_date, is_current, version,
                  source_session, source_agent, created_at
           FROM memories ORDER BY created_at"""
    ).fetchall()

    nodes = []
    for r in rows:
        nodes.append({
            "id": r["id"],
            "content": r["content"],
            "category": r["category"],
            "confidence": r["confidence"],
            "documentDate": r["document_date"],
            "eventDate": r["event_date"],
            "isCurrent": bool(r["is_current"]),
            "version": r["version"],
            "session": r["source_session"],
            "agent": r["source_agent"],
            "createdAt": r["created_at"],
        })

    # Get all relations
    rel_rows = conn.execute(
        """SELECT mr.from_memory, mr.to_memory, mr.relation, mr.context,
                  ms.content as source_content, mt.content as target_content
           FROM memory_relations mr
           JOIN memories ms ON mr.from_memory = ms.id
           JOIN memories mt ON mr.to_memory = mt.id"""
    ).fetchall()

    edges = []
    for r in rel_rows:
        edges.append({
            "source": r["from_memory"],
            "target": r["to_memory"],
            "type": r["relation"],
            "context": r["context"],
            "sourceContent": r["source_content"],
            "targetContent": r["target_content"],
        })

    conn.close()
    return {"nodes": nodes, "edges": edges}


@app.get("/api/stats")
async def stats():
    return engine.get_stats()


@app.get("/api/history/{entity_name}")
async def history(entity_name: str):
    return {"history": engine.get_history(entity_name)}


@app.get("/api/profile/{entity_name}")
async def profile(entity_name: str):
    return engine.get_profile(entity_name)


@app.get("/api/entities")
async def entities():
    """Return unique entity names."""
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    # entity_name may not exist as column; extract from content
    rows = conn.execute(
        "SELECT DISTINCT entity_name FROM profiles ORDER BY entity_name"
    ).fetchall()
    conn.close()
    return {"entities": [r[0] for r in rows]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8642)
