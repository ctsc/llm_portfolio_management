"""Build a ProviderBundle from config: synthetic world, fixture replay, or live services."""

from __future__ import annotations

from pathlib import Path

from portstudy.config import StudyConfig, repo_root
from portstudy.data.providers.base import ProviderBundle
from portstudy.data.providers.http import FixtureHttp, HttpClient, LiveHttp
from portstudy.data.providers.synthetic import (
    SyntheticWorld,
    load_or_generate_world,
    synthetic_bundle,
)
from portstudy.paths import synthetic_dir


def synthetic_world(cfg: StudyConfig, folder: Path | None = None) -> SyntheticWorld:
    target = folder or synthetic_dir()
    return load_or_generate_world(cfg.synthetic, target, first_week=cfg.synthetic.first_week)


def build_providers(
    cfg: StudyConfig,
    synthetic: bool = False,
    world: SyntheticWorld | None = None,
    fixture_mode: bool = False,
) -> ProviderBundle:
    """Return the configured providers. Live adapters need API keys in the environment."""
    if synthetic:
        return synthetic_bundle(world or synthetic_world(cfg))
    return build_real_providers(cfg, fixture_mode=fixture_mode)


def build_real_providers(cfg: StudyConfig, fixture_mode: bool = False) -> ProviderBundle:
    # Imported lazily so the core never needs these modules to be importable in tests.
    from portstudy.data.providers.alpaca import AlpacaNewsProvider, AlpacaPriceProvider
    from portstudy.data.providers.finnhub import FinnhubEarningsProvider
    from portstudy.data.providers.fred import FredRateProvider
    from portstudy.data.providers.wikipedia import WikipediaMembersProvider
    from portstudy.data.providers.yfinance_provider import (
        FixtureFetch,
        YFinancePriceProvider,
        YFinanceReferenceProvider,
    )

    d = cfg.data
    limits = d.rate_limits

    def http(per_minute: int) -> HttpClient:
        live = LiveHttp(per_minute, d.retries.attempts, d.retries.backoff_seconds)
        if fixture_mode:
            return FixtureHttp(repo_root() / d.fixture_dir, recorder=None)
        return live

    members = WikipediaMembersProvider(http(limits.yfinance_per_minute))
    earnings = FinnhubEarningsProvider(http(limits.finnhub_per_minute))
    rates = FredRateProvider(http(limits.fred_per_minute))

    if d.price_provider == "alpaca":
        prices = AlpacaPriceProvider(http(limits.alpaca_per_minute))
    elif d.price_provider == "yfinance":
        fetch = FixtureFetch(repo_root() / d.fixture_dir) if fixture_mode else None
        prices = YFinancePriceProvider(fetch.history) if fetch else YFinancePriceProvider()  # type: ignore[assignment]
    else:
        raise ValueError(f"unknown price provider {d.price_provider}")

    if d.news_provider == "alpaca":
        news = AlpacaNewsProvider(http(limits.alpaca_per_minute))
    else:
        raise ValueError(f"unknown news provider {d.news_provider}")

    fetch_info = FixtureFetch(repo_root() / d.fixture_dir) if fixture_mode else None
    reference = (
        YFinanceReferenceProvider(fetch_info.info, earnings=earnings)
        if fetch_info
        else YFinanceReferenceProvider(earnings=earnings)
    )
    return ProviderBundle(
        prices=prices, news=news, reference=reference, rates=rates, members=members
    )
