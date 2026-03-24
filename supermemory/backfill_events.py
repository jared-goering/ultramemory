#!/usr/bin/env python3
"""Backfill event_mentions and event_clusters from existing source_chunks.

Reads all source_chunks, calls extract_events for each that hasn't been processed yet.

Usage:
    python -m supermemory.backfill_events [--db PATH] [--verbose] [--dry-run]
"""

import argparse
import sys

from supermemory.config import get_config
from supermemory.engine import MemoryEngine


def backfill(db_path: str | None = None, verbose: bool = False, dry_run: bool = False) -> int:
    """Process all source_chunks that don't yet have event_mentions.

    Returns the number of events extracted.
    """
    cfg = get_config()
    db_path = db_path or cfg["db_path"]

    engine = MemoryEngine(db_path=db_path)
    conn = engine._conn()

    try:
        # Find chunks that already have event_mentions
        processed_chunks = {
            row["source_chunk_id"]
            for row in conn.execute(
                "SELECT DISTINCT source_chunk_id FROM event_mentions "
                "WHERE source_chunk_id IS NOT NULL"
            ).fetchall()
        }

        # Get all source_chunks
        all_chunks = conn.execute(
            "SELECT id, content, session_key, document_date FROM source_chunks "
            "ORDER BY created_at ASC"
        ).fetchall()
    finally:
        conn.close()

    to_process = [c for c in all_chunks if c["id"] not in processed_chunks]

    if verbose:
        print(
            f"Found {len(all_chunks)} total chunks, "
            f"{len(processed_chunks)} already processed, "
            f"{len(to_process)} to backfill",
            file=sys.stderr,
        )

    total_events = 0

    for i, chunk in enumerate(to_process):
        chunk_id = chunk["id"]
        content = chunk["content"]
        session_key = chunk["session_key"] or "unknown"
        document_date = chunk["document_date"]

        if not content or len(content.strip()) < 50:
            if verbose:
                print(
                    f"  [{i + 1}/{len(to_process)}] Skipping short chunk {chunk_id[:8]}",
                    file=sys.stderr,
                )
            continue

        if dry_run:
            print(
                f"  [{i + 1}/{len(to_process)}] Would process chunk {chunk_id[:8]} ({len(content)} chars)",
                file=sys.stderr,
            )
            continue

        if verbose:
            print(
                f"  [{i + 1}/{len(to_process)}] Processing chunk {chunk_id[:8]}...",
                file=sys.stderr,
                end=" ",
            )

        try:
            events = engine.extract_events(
                text=content,
                session_key=session_key,
                chunk_id=chunk_id,
                document_date=document_date,
            )
            total_events += len(events)
            if verbose:
                print(f"{len(events)} events", file=sys.stderr)
                for ev in events:
                    print(f"    * [{ev['event_type']}] {ev['summary'][:70]}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing chunk {chunk_id[:8]}: {e}", file=sys.stderr)

    return total_events


def main():
    parser = argparse.ArgumentParser(description="Backfill events from existing source_chunks")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite database")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    args = parser.parse_args()

    total = backfill(db_path=args.db, verbose=args.verbose, dry_run=args.dry_run)
    print(f"Backfill complete: {total} events extracted")


if __name__ == "__main__":
    main()
