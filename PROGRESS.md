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

## Phase 2: Synthetic world and data layer

Built:
- `data/providers/base.py`: `PriceProvider`, `NewsProvider`, `ReferenceProvider`, `RateProvider`, `MembersProvider`, `ProviderBundle`, fixed column names.
- `data/providers/synthetic.py`: seeded factor-model world (120 tickers, 400 history days, 40 future weeks) with planted momentum, planted news sentiment effect, split, dividend, delisting, halt, extreme move, zero price, stale price, duplicate rows, short history, no-news weeks, and post-cutoff articles. ETF proxies SPY, IWM, IWD, IWF, QUAL.
- `scripts/make_synthetic_world.py` writes the world and the synthetic universe to `data/synthetic/`.
- Real adapters behind the same interfaces, mock-tested only: `alpaca.py` (bars raw plus adjusted, news), `yfinance_provider.py` (prices, info, fallback sectors), `finnhub.py` (earnings), `fred.py` (DTB3), `wikipedia.py` (S&P 500 constituents). Shared `http.py` with tenacity retries, a token-bucket rate limit from config, and `FixtureHttp` replay. `factory.py` builds the bundle from config.
- `data/checks.py`: missing days, nonpositive price, extreme move (split-aware), duplicate rows, stale price, no news, news after cutoff, short history, stopped trading; each maps to a config rule and is logged.
- `data/ledger.py`: append-only raw price ledger and corporate actions, `ledger_mismatch` warning on differing provider values.
- `data/snapshot.py`: `make_snapshot` (strict cutoff filter, dedupe, token budget, article_count, earnings_in_window, immutable versions, manifest hashes, ledger append), `read_snapshot`.
- `data/universe.py`: sector-balanced top-9-per-sector plus one, `extra_tickers` replacement, `config/universe.csv` writer with a printed sector table.

Tests: 31 unit tests plus `tests/integration/test_snapshots_30_weeks.py` (30 consecutive snapshots, only planted issues, ledger has no duplicate days). `ruff check`, `mypy src` clean.

Fixes made while finishing the phase:
- `.gitignore` used `data/`, which also ignored `src/portstudy/data/`; changed to `/data/`.
- `_forward_fill` broadcast bug on halted stocks (fired on the Good Friday test).
- Removed-stock test now picks a week after the planted delisting instead of a fixed index.

UNVERIFIED items (Phase 17 smoke test):
- `alpaca.py`: `AlpacaPriceProvider` (`/v2/stocks/bars`, adjustment raw and all, `/v1/corporate-actions`), `AlpacaNewsProvider` (`/v1beta1/news`).
- `yfinance_provider.py`: `YFinancePriceProvider`, `YFinanceReferenceProvider` (history, info fields `marketCap`, `sector`, `longName`).
- `finnhub.py`: `FinnhubEarningsProvider` (`/calendar/earnings`).
- `fred.py`: `FredRateProvider` (`/fred/series/observations`, series DTB3).
- `wikipedia.py`: `WikipediaMembersProvider` (table id `constituents`).
