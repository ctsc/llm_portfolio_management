"""Make one real call per service with a tiny request and print status, cost, and latency.

Run this on the desktop after the keys are in `.env` to expose every UNVERIFIED item (the
gateway models need the GSU network or VPN). Nothing here is called by tests or by the
pipeline. Each check is independent; a missing key or package is reported and skipped, never
fatal to the others.

Usage: python scripts/smoke_real_services.py [--only alpaca,fred,...] [--ticker AAPL]
Keys (environment or .env): ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, FINNHUB_API_KEY,
FRED_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, ARCTIC_LITELLM_API_KEY.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from portstudy.config import (
    LlmSettings,
    ModelRegistry,
    ModelSpec,
    StudyConfig,
    load_model_registry,
    load_study_config,
)
from portstudy.data.providers.http import HttpClient
from portstudy.llm.cost import CostTable
from portstudy.llm.prompts import assemble_prompt, load_manifest
from portstudy.llm.providers.base import BatchItem, LlmProvider, RawResult, make_custom_id
from portstudy.llm.runner import settings_for_model
from portstudy.llm.validate import parse_response
from portstudy.paths import outputs_dir


def time_gateway(
    cfg: StudyConfig,
    registry: ModelRegistry,
    n: int,
    out_path: Path,
    providers: dict[str, LlmProvider] | None = None,
) -> pd.DataFrame:
    """DECISION(G15): send `n` real rating requests per gateway model at its configured
    concurrency, cycling through the enabled prompt configurations, and time each call
    client-side and via the gateway's x-litellm-response-duration-ms header. Writes one CSV row
    per request to `out_path` and returns the frame."""
    from concurrent.futures import ThreadPoolExecutor

    from portstudy.llm.providers.litellm import LiteLLMProvider

    manifest = load_manifest()
    configs = list(manifest.enabled())
    rows: list[dict[str, object]] = []
    for spec in registry.enabled():
        if spec.kind != "open_weight" or spec.serving != "gateway":
            continue
        settings = settings_for_model(cfg, spec)
        provider = (providers or {}).get(spec.id) or LiteLLMProvider(
            spec.id,
            spec.provider_model,
            cfg.llm.gateway.base_url,
            "ARCTIC_LITELLM_API_KEY",
            timeout=settings.request_timeout_seconds,
            features_frozen=cfg.llm.gateway.features_frozen,
        )
        items: list[BatchItem] = []
        for index in range(n):
            conf = configs[index % len(configs)]
            prompt = assemble_prompt(
                conf.id,
                {"ticker": "SMOK", "name": "Example Co", "sector": "Information Technology"},
                "Example Co reports a routine quarterly update.",
                cfg,
                manifest,
            )
            items.append(
                BatchItem(
                    make_custom_id(spec.id, conf.id, "smoke", "SMOK"),
                    spec.id,
                    conf.id,
                    "smoke",
                    "SMOK",
                    prompt,
                )
            )
        workers = max(1, cfg.llm.concurrency.get(spec.id, 4))

        def call(
            pair: tuple[int, BatchItem],
            _provider: LlmProvider = provider,
            _settings: LlmSettings = settings,
            _spec: ModelSpec = spec,
        ) -> dict[str, object]:
            index, item = pair
            started = time.perf_counter()
            try:
                raw = _provider.complete(item, _settings)
                error: str | None = raw.error
            except Exception as exc:
                raw = RawResult(custom_id=item.custom_id, text="", error=str(exc)[:500])
                error = raw.error
            headers = raw.gateway_headers
            gateway_ms = headers.get("x-litellm-response-duration-ms")
            ratelimit = {k: v for k, v in headers.items() if k.startswith("x-ratelimit-")}
            parsed = parse_response(
                raw.text,
                cfg.llm.rationale_token_cap,
                raw.stop_reason,
                reasoning_text=raw.reasoning_text,
            )
            return {
                "model": _spec.id,
                "prompt_id": item.config_id,
                "request_index": index,
                "client_seconds": time.perf_counter() - started,
                "gateway_ms": float(gateway_ms) if gateway_ms else float("nan"),
                "prompt_tokens": raw.tokens_in,
                "completion_tokens": raw.tokens_out,
                "reasoning_tokens": raw.tokens_reasoning,
                "finish_reason": raw.finish_reason or "",
                "valid": bool(parsed.valid) and error is None,
                "error": error or "",
                "send_seed": raw.send_seed,
                "json_mode": raw.json_mode,
                "deployment_id": raw.deployment_id,
                "gateway_version": raw.gateway_version,
                "response_cost": headers.get("x-litellm-response-cost", ""),
                "ratelimit": json.dumps(ratelimit, sort_keys=True),
            }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            model_rows = list(pool.map(call, list(enumerate(items))))
        rows.extend(model_rows)
        frame = pd.DataFrame(model_rows)
        client = frame["client_seconds"]
        gateway = frame["gateway_ms"].dropna()
        limits: dict[str, str] = {}
        for entry in model_rows:
            limits.update(
                {
                    k: v
                    for k, v in json.loads(str(entry["ratelimit"])).items()
                    if k.startswith("x-ratelimit-limit-")
                }
            )
        print(
            f"{spec.id}: client mean {client.mean():.2f}s p50 {client.quantile(0.5):.2f}s "
            f"p95 {client.quantile(0.95):.2f}s; gateway mean {gateway.mean():.0f}ms "
            f"p50 {gateway.quantile(0.5):.0f}ms p95 {gateway.quantile(0.95):.0f}ms"
            if not gateway.empty
            else f"{spec.id}: client mean {client.mean():.2f}s (no gateway timings)"
        )
        print(
            f"  ratelimit limits: {limits or 'none seen'}; "
            f"invalid rate {1.0 - frame['valid'].mean():.2f}; "
            f"finish_reason=length share {(frame['finish_reason'] == 'length').mean():.2f}"
        )
        print(
            f"  tokens prompt mean {frame['prompt_tokens'].mean():.0f} "
            f"max {frame['prompt_tokens'].max()}; completion mean "
            f"{frame['completion_tokens'].mean():.0f} max {frame['completion_tokens'].max()}; "
            f"reasoning mean {frame['reasoning_tokens'].mean():.0f} "
            f"max {frame['reasoning_tokens'].max()}"
        )
        print(f"  gateway accepted seed={provider.send_seed} json_mode={provider.json_mode}")
    frame = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def alpaca_check(
    cfg: StudyConfig, http: HttpClient, ticker: str, start: dt.date, end: dt.date
) -> str:
    """Request daily bars and news for one ticker with the configured feed (DECISION(G16),
    review C11). Raises AssertionError unless the last returned bar is the last trading day in
    [start, end]; a refused feed surfaces the provider's own error through `timed`."""
    from portstudy import calendar as cal
    from portstudy.data.providers.alpaca import AlpacaNewsProvider, AlpacaPriceProvider

    prices = AlpacaPriceProvider(http, feed=cfg.data.alpaca_feed).get_prices([ticker], start, end)
    news = AlpacaNewsProvider(http).get_news(
        [ticker],
        dt.datetime.combine(start, dt.time(), tzinfo=dt.UTC),
        dt.datetime.combine(end, dt.time(), tzinfo=dt.UTC),
    )
    days = cal.trading_days(start, end)
    last_day = days[-1] if days else end
    last_bar = prices["date"].max() if len(prices) else None
    assert last_bar == last_day, (
        f"alpaca returned no bar for the last trading day {last_day} "
        f"(last bar: {last_bar}, feed: {cfg.data.alpaca_feed})"
    )
    volume = float(prices.loc[prices["date"] == last_bar, "volume"].iloc[-1])
    return (
        f"feed={cfg.data.alpaca_feed}, {len(prices)} bars, last bar {last_bar} "
        f"vol={volume:.0f}, {len(news)} articles"
    )


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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--time-gateway",
        type=int,
        metavar="N",
        help="send N rating requests per enabled gateway model and write gateway_timing.csv",
    )
    args = parser.parse_args()
    load_dotenv()
    cfg = load_study_config()
    registry = load_model_registry()
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    if args.dry_run:
        from portstudy.llm.providers.litellm import LiteLLMProvider

        manifest = load_manifest()
        prompt = assemble_prompt(
            "bare",
            {"ticker": args.ticker, "name": "Example Co", "sector": "Information Technology"},
            "Example Co reports a routine quarterly update.",
            cfg,
            manifest,
        )
        for spec in registry.enabled():
            if spec.kind != "open_weight" or spec.serving != "gateway":
                continue
            if wanted and spec.id not in wanted and spec.provider not in wanted:
                continue
            provider = LiteLLMProvider(
                spec.id,
                spec.provider_model,
                cfg.llm.gateway.base_url,
                "ARCTIC_LITELLM_API_KEY",
            )
            item = BatchItem(
                make_custom_id(spec.id, "bare", "smoke", args.ticker),
                spec.id,
                "bare",
                "smoke",
                args.ticker,
                prompt,
            )
            print(
                json.dumps(
                    {
                        "model": spec.id,
                        "endpoint": cfg.llm.gateway.base_url,
                        "payload": provider.build_payload(item, settings_for_model(cfg, spec)),
                    },
                    sort_keys=True,
                )
            )
        return

    if args.time_gateway:
        out = outputs_dir() / "gateway_timing.csv"
        frame = time_gateway(cfg, registry, args.time_gateway, out)
        print(f"wrote {out} ({len(frame)} requests)")
        return

    end = dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=10)

    def want(name: str) -> bool:
        return not wanted or name in wanted

    from portstudy.data.providers.http import LiveHttp

    http = LiveHttp(per_minute=30)

    if want("alpaca"):
        timed("alpaca", lambda: alpaca_check(cfg, http, args.ticker, start, end))

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

            provider = make_provider(spec, gateway=cfg.llm.gateway)
            item = BatchItem(
                make_custom_id(spec.id, "bare", "smoke", args.ticker),
                spec.id,
                "bare",
                "smoke",
                args.ticker,
                prompt,
            )
            raw = provider.complete(item, settings_for_model(cfg, spec))
            parsed = parse_response(
                raw.text,
                cfg.llm.rationale_token_cap,
                raw.stop_reason,
                reasoning_text=raw.reasoning_text,
            )
            cost = table.cost_usd(
                spec.id, raw.tokens_in, raw.tokens_cached, raw.tokens_out, batch=False
            )
            return (
                f"version={raw.model_version} valid={parsed.valid} likely={parsed.likely} "
                f"tokens={raw.tokens_in}/{raw.tokens_out} cost={cost:.5f} USD"
            )

        timed(spec.id, call)

    print()
    print("Fix every FAILED line and every UNVERIFIED item in plans/PROGRESS.md before the pilot.")


if __name__ == "__main__":
    main()
