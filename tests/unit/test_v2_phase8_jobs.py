from __future__ import annotations

import pandas as pd

from portstudy.optimize import _cov_mu_sha256
from portstudy.optimize.black_litterman import Posterior
from portstudy.pipeline.fallback import FallbackPlan


def test_source_fallback_status_applies_to_every_strategy() -> None:
    plan = FallbackPlan(
        week_id="2026-W10",
        statuses={"stub-ow-a__bare": "fallback"},
        reasons={"stub-ow-a__bare": "invalid_rate"},
    )

    assert plan.status_for("mvo__stub-ow-a__bare") == "fallback"
    assert plan.status_for("cvar__stub-ow-a__bare") == "fallback"
    assert plan.reason_for("topk__stub-ow-a__bare") == "invalid_rate"
    assert plan.status_for("mvo__Q0") == "normal"
    assert plan.status_for("HRP") == "normal"


def test_covariance_of_mean_hash_is_stable_and_sensitive_to_matrix() -> None:
    index = pd.Index(["AAA", "BBB"])
    posterior = Posterior(
        mu_bl=pd.Series([0.01, 0.02], index=index),
        tilt=pd.Series([0.001, 0.002], index=index),
        n_views=1,
        cov_mu=pd.DataFrame([[0.1, 0.01], [0.01, 0.2]], index=index, columns=index),
    )

    first = _cov_mu_sha256(posterior)
    assert first == _cov_mu_sha256(posterior)
    posterior.cov_mu.iloc[0, 0] += 1e-12
    assert first != _cov_mu_sha256(posterior)
