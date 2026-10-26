"""Make one real call per service with a tiny request and print status, cost, and latency.

Phase 17 B4: run this on the desktop after the keys are in `.env` to expose every UNVERIFIED
item. Nothing here is called by tests or by the pipeline. Each check is independent; a
missing key or package is reported and skipped, never fatal to the others.

Usage: python scripts/smoke_real_services.py [--only alpaca,fred,...] [--ticker AAPL]
Keys (environment or .env): ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, FINNHUB_API_KEY,
FRED_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY. Ollama needs a running local server.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from collections.abc import Callable
from pathlib import Path

from portstudy.config import load_model_registry, load_study_config
from portstudy.llm.cost import CostTable
from portstudy.llm.prompts import assemble_prompt, load_manifest
from portstudy.llm.providers.base import BatchItem, make_custom_id
from portstudy.llm.validate import parse_response


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def timed(name: str, fn: Callable[[], str]) -> None:
    start = time.perf_counter()
    try:
        detail = fn()
        status = "ok"
    except Exception as exc:  # report and move on to the next service
        detail = f"{type(exc).__name__}: {exc}"
        status = "FAILED"
    print(f"{name:12s} {status:7s} {time.perf_counter() - start:6.1f}s  {detail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()
    load_dotenv()
    cfg = load_study_config()
    registry = load_model_registry()
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    end = dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=10)

    def want(name: str) -> bool:
        return not wanted or name in wanted

    from portstudy.data.providers.http import LiveHttp

    http = LiveHttp(per_minute=30)

    if want("alpaca"):

        def alpaca() -> str:
            from portstudy.data.providers.alpaca import AlpacaNewsProvider, AlpacaPriceProvider

            prices = AlpacaPriceProvider(http).get_prices([args.ticker], start, end)
            news = AlpacaNewsProvider(http).get_news(
                [args.ticker],
                dt.datetime.combine(start, dt.time(), tzinfo=dt.UTC),
                dt.datetime.combine(end, dt.time(), tzinfo=dt.UTC),
            )
            return f"{len(prices)} bars, {len(news)} articles"

        timed("alpaca", alpaca)

    if want("yfinance"):

        def yfinance() -> str:
            from portstudy.data.providers.yfinance_provider import (
                YFinancePriceProvider,
                YFinanceReferenceProvider,
            )

            prices = YFinancePriceProvider().get_prices([args.ticker], start, end)
            ref = YFinanceReferenceProvider().get_reference([args.ticker], end)
            return f"{len(prices)} bars, sector={ref['sector'].iloc[0] if len(ref) else None}"

        timed("yfinance", yfinance)

    if want("finnhub"):

        def finnhub() -> str:
            from portstudy.data.providers.finnhub import FinnhubEarningsProvider

            frame = FinnhubEarningsProvider(http).get_earnings(
                [args.ticker], end, end + dt.timedelta(days=120)
            )
            return f"{len(frame)} earnings dates"

        timed("finnhub", finnhub)

    if want("fred"):

        def fred() -> str:
            from portstudy.data.providers.fred import FredRateProvider

            rates = FredRateProvider(http).get_rates(start, end)
            last = rates["rate"].iloc[-1] if len(rates) else None
            return f"{len(rates)} DTB3 observations, last={last}"

        timed("fred", fred)

    if want("wikipedia"):

        def wikipedia() -> str:
            from portstudy.data.providers.wikipedia import WikipediaMembersProvider

            members = WikipediaMembersProvider(http).get_members(end)
            return f"{len(members)} S&P 500 members, {members['sector'].nunique()} sectors"

        timed("wikipedia", wikipedia)

    # One tiny prompt per enabled model, synchronous (no batch), to check settings and parsing.
    manifest = load_manifest()
    prompt = assemble_prompt(
        "bare",
        {"ticker": args.ticker, "name": "Example Co", "sector": "Information Technology"},
        "Example Co reports a routine quarterly update.",
        cfg,
        manifest,
    )
    table = CostTable(registry)
    for spec in registry.enabled():
        if not want(spec.provider) and not want(spec.id):
            continue

        def call(spec=spec) -> str:  # type: ignore[no-untyped-def]
            from portstudy.llm.providers.factory import make_provider

            provider = make_provider(spec)
            item = BatchItem(
                make_custom_id(spec.id, "bare", "smoke", args.ticker),
                spec.id,
                "bare",
                "smoke",
                args.ticker,
                prompt,
            )
            raw = provider.complete(item, cfg.llm.settings.get(spec.id, spec.settings))
            parsed = parse_response(raw.text, cfg.llm.rationale_token_cap, raw.stop_reason)
            cost = table.cost_usd(
                spec.id, raw.tokens_in, raw.tokens_cached, raw.tokens_out, batch=False
            )
            return (
                f"version={raw.model_version} valid={parsed.valid} score={parsed.score} "
                f"tokens={raw.tokens_in}/{raw.tokens_out} cost={cost:.5f} USD"
            )

        timed(spec.id, call)

    print()
    print("Fix every FAILED line and every UNVERIFIED item listed in PROGRESS.md before the pilot.")


if __name__ == "__main__":
    main()
