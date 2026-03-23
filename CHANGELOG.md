# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-23

### Added
- Proper Python package structure (`supermemory/` directory)
- `supermemory serve` CLI command for starting the API server
- GitHub Actions CI (Python 3.10-3.12)
- CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md
- Pre-commit hooks (ruff linting + formatting)
- Issue and PR templates

### Changed
- Moved all modules into `supermemory/` package (was flat root files)
- `sentence-transformers` is now an optional dependency: `pip install supermemory[local]`
- Replaced personal data in tests/docs with generic examples

### Removed
- Machine-specific files (launchd plists, bulk logs)
- Standalone `supermemory-serve` entry point (use `supermemory serve` instead)

### Security
- Scrubbed personal information from test fixtures and documentation
- Added git history rewrite to remove PII from past commits

## [0.1.0] - 2026-03-23

### Added
- Core memory engine with SQLite storage
- Atomic fact extraction via LLM (litellm multi-provider)
- Relational versioning (updates, extends, contradicts, supports, derives)
- Temporal grounding (documentDate, eventDate, as-of-date queries)
- Semantic search with local embeddings (sentence-transformers)
- Entity profiles and version history
- FastAPI server with REST API
- CLI tools (init, ingest, search, stats, history, profile)
- Live ingest pipeline for OpenClaw session files
- Deduplication (exact + semantic at 0.97+ cosine similarity)
- React + Next.js visualization UI with 3D force graph
