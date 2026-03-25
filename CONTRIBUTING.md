# Contributing to Ultramemory

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/jared-goering/ultramemory.git
cd ultramemory

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in dev mode with all extras
pip install -e ".[local]"
pip install ruff pytest pre-commit

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
pytest tests/ -v
```

Tests mock all LLM and embedding calls, so no API keys are needed.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix what's possible
ruff check --fix .

# Format code
ruff format .
```

Pre-commit hooks run these automatically on each commit.

## Submitting Changes

1. Fork the repo and create a feature branch from `main`
2. Make your changes
3. Add or update tests as needed
4. Run `pytest tests/ -v` and `ruff check .` to verify
5. Commit with a clear message describing what changed
6. Open a Pull Request against `main`

## What to Work On

Check [open issues](https://github.com/jared-goering/ultramemory/issues) for ideas. Issues labeled `good first issue` are a great starting point.

## Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS

## Security

If you find a security vulnerability, please report it privately. See [SECURITY.md](SECURITY.md).
