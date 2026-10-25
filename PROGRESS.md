# PROGRESS.md: agent build log

Each entry: what was built, tests passing, anything skipped and why.

## Phase 1: Scaffold, config, and utilities

Built:
- `pyproject.toml` with dependency groups `ml`, `api`, `data`, `dev`; core installs without extras.
- `config/study.yaml` (every section and every decision from devin.md 0.3) and `config/models.yaml` (model registry with stub models for tests).
- `src/portstudy/config.py`: strict pydantic models, unknown keys rejected, decisions carry `id`, `value`, `status`.
- `paths.py`, `hashing.py`, `io.py` (atomic writes, `%.12g` floats), `logging_setup.py` (JSON lines, ET timestamps), `types.py`, `calendar.py` (built-in NYSE table 2024 to 2027 with early closes, `pandas_market_calendars` when installed).

Tests: 14 unit tests passing (`tests/unit/test_phase1_scaffold.py`). `ruff check`, `mypy src` clean.

Notes:
- Python 3.11 installed via `uv`; the venv lives at `.venv`.
- Holiday table hand-entered from the NYSE schedule; `pandas_market_calendars` (data extra) takes over when installed.

UNVERIFIED items (to be smoke-tested in Phase 17):
- `config/models.yaml`: Ollama tags for `qwen3.5:4b`, `phi4-mini`, `llama3.2:3b`; dated snapshot IDs for `claude-haiku-5.5` and `gpt-6-luna`; all prices are placeholders.
