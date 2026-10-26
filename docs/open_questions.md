# Open questions and study decisions

A mirror of the decision table in `PLAN.md` section 0.3, with the place each default lives in
`config/study.yaml` (or `config/models.yaml`) and the code locations tagged `DECISION(<id>)`.
Rows marked `provisional` may change after the real prompt test runs (Phase 17 D); everything
else is confirmed.

Rule: changing any default after the live run starts must be logged in `PROGRESS.md` with the
date and reason, committed as a new version, and flagged in the analysis. Code never changes a
decision value.

| ID | Setting | Value | Status | Config location | Code tags |
|---|---|---|---|---|---|
| D1 | models | local: Qwen 3.5 4B (thinking off), Phi-4 Mini; fallback Llama 3.2 3B; API: Claude Haiku 5.5, GPT-6 Luna (reasoning none) | confirmed | `config/models.yaml` (`models`), `decisions[D1]` | `llm/providers/openai_batch.py` (reasoning none) |
| D3 | budget | goal 50 USD; warn at 50; `--allow-over-budget` to pass it | provisional | `llm.cost_goal_usd`, `llm.cost_warn_usd`, `decisions[D3]` | `llm/cost.py`, `scripts/estimate_cost.py` |
| D4 | temperature | 0 for all models | confirmed | `llm.temperature`, per-model `settings.temperature`, `decisions[D4]` | provider payloads in `llm/providers/*` |
| B1 | prompt configurations | bare, skills, placebo (more skills files may be added as new entries) | confirmed | `config/prompts_manifest.yaml`, `decisions[B1]` | `llm/prompts.py` |
| B2 | stock list | sector balanced, drawn from the S&P 500 | confirmed | `universe.source`, `universe.per_sector`, `decisions[B2]` | `data/universe.py` |
| B4 | universe size | 100 | confirmed | `universe.size`, `decisions[B4]` | `data/universe.py` |
| A2 | start | first live week is the week after the pilot passes | confirmed | `timing.live_start_week`, `decisions[A2]` | `pipeline/context.py` |
| A3 | live length | until `timing.live_end_date`, no later than the April deadline | confirmed | `timing.live_end_date`, `decisions[A3]` | `llm/cost.py` (weeks remaining) |
| A4 | closed Friday | the last prior trading day is the cutoff | confirmed | `decisions[A4]` | `calendar.py` |
| A6 | cold start | first 5 live weeks use IC 0.02 | confirmed | `views.cold_start_weeks`, `views.cold_start_ic`, `decisions[A6]` | `views/ic.py` |
| C1 | prompt fields | ticker, company name, sector, news | confirmed | `llm.prompt_fields`, `decisions[C1]` | `llm/prompts.py` |
| C2 | stock with no news | still call the model with the fixed sentence; no tools or web search | confirmed | `llm.no_news_sentence`, `decisions[C2]` | `llm/prompts.py` |
| C4 | skills file token cap | 6000 | provisional | `llm.skills_token_cap`, `decisions[C4]` | `scripts/check_placebo_length.py` |
| C7 | rationale token cap | 150 | provisional | `llm.rationale_token_cap`, `decisions[C7]` | `llm/validate.py` |
| E2 | IC used to scale the LLM view | IC of the residualized signal | confirmed | `views.ic_signal_for_llm`, `decisions[E2]` | `views/residualize.py`, `optimize/__init__.py` |
| E3 | combined score for H2 | ranked Black-Litterman tilt | confirmed | `eval.combined_signal`, `decisions[E3]` | `pipeline/jobs.py` (`update_ic_history`), `eval/hypotheses.py` |
| E4 | Kronos input prices | adjusted OHLCV | confirmed | `baseline.kronos.input_prices`, `decisions[E4]` | `baseline/combine.py` |
| E5 | market-cap weights | over the study's stocks only, renormalized | confirmed | `risk.market_cap_weights`, `decisions[E5]` | `risk/prior.py` |
| G3 | real H3 effect | paired weekly test passes after Holm, placebo as control | confirmed | `eval.holm_families`, `decisions[G3]` | `eval/hypotheses.py` |
| G4 | H4 splits | market up/down, volatility, earnings week | confirmed | `eval.h4_splits`, `decisions[G4]` | `eval/hypotheses.py` |
| G5 | factor regression | daily returns | confirmed | `eval.factor_frequency`, `decisions[G5]` | `eval/factors.py` |
| G6 | shuffled-ratings test | every portfolio plus Q0, 500 shuffles each | confirmed | `eval.shuffle.n_shuffles`, `decisions[G6]` | `eval/shuffle.py` |
| N1 | failure rule | invalid rate above 0.15, or local job or batch not finished by the lock | confirmed | `llm.invalid_rate_threshold`, `pipeline.lock_margin_minutes`, `decisions[N1]` | `llm/runner.py` (`assess_run`), `pipeline/jobs.py`, `pipeline/fallback.py` |
| N2 | weekly trigger | one hand-run script, no scheduler | confirmed | `decisions[N2]` | `scripts/run_weekend.py` |
| N3 | storage and backup | CSV files; dated folders downloaded weekly | confirmed | `backup.*`, `decisions[N3]` | `backup.py` |

## Provisional items the researcher settles in Phase 17

- D3 budget: rerun `scripts/estimate_cost.py` after the real prompt test runs and set
  `llm.cost_goal_usd`.
- C4 skills cap and C7 rationale cap: check the real skills file with
  `scripts/check_placebo_length.py`; raise `max_output_tokens` if test runs show
  `invalid_truncated`.

## Items marked UNVERIFIED until the real-key smoke test

Listed per phase in `PROGRESS.md`; `scripts/smoke_real_services.py` exercises each one.
