# LLM research knowledge base

Last updated: 2026-11-17. Carter Tierney's research notes, kept here so findings and decisions
survive chat compaction. Everything in this folder reflects the current plan unless a section is
labeled as dropped, superseded or history.

## Current project status (2026-11-11)

- Study: "Specialized LLM Skills Across Fixed Quantitative Portfolio Optimization Strategies" (MS
  project, advisor Dr. Parag Tamhankar).
- Research proposal and ARCTIC request updated after Parag's feedback (copies in
  `current_documents/`).
- ARCTIC allocation (granted 2026-11-16): the default startup allocation of 1,000 GPU and 10,000 CPU service units, so the study uses 100 stocks. ARCTIC request (revised 2026-11-12): GPU hours for gateway LLM calls only (2,730 for 300 stocks, or
  910 for 100), on
  the two models already hosted (Gemma 4 31B, MiniMax M2.7). No CPU hours, storage or deployments.
- Everything except the LLM calls runs on Carter's desktop; benchmarks on 2026-11-10 confirmed it
  fits.
- Evaluation: standard quant tests on saved data only. The shuffle test is dropped.
- ARCTIC review (2026-11-11): the wiki and both model cards were read (`09`). The gateway is an alpha
  service; access now goes through Parag's Elpis request (2026-11-12). How it is metered and the
  GPU-second basis are still unknown. MiniMax M2.7 always thinks, so its output cap and the request's
  token figures were changed (request revised 2026-11-12, finalized 2026-11-16). Carter needs access
  by the end of November.
- Review 5 (2026-11-11, `plans/REVIEW5_fable_2026-11-11.md`): the LLM view becomes neutral at a rating
  of 0 (E7), Mean-CVaR and Resampled MVO are rescaled before the freeze (S8), the stop rule is fixed
  (G1b), and several gateway, data and simulator bugs are fixed in PLAN4. Power numbers: the smallest
  detectable weekly IC is about 0.034 at 300 stocks and 0.059 at 100.
- Andalib replied 2026-11-12 (`09` section G): no formal gateway-only process yet, metering and
  limits being decided, seed passed through but outputs not identical, no changelog. Added the
  pilot repeat check (G20) and weekly canaries (G21); temperature 0 confirmed; request revised.
- Andalib follow-ups (to 2026-11-16): Parag submits through Elpis (no Hydra); MiniMax reasoning in
  `reasoning_content` with `reasoning_tokens` in `usage`; no model version from the gateway, but
  deployment ID and gateway version headers and `/v1/model/info` can be logged. Request finalized
  2026-11-16.
- Next: Parag submits in Elpis, ask GSU IT about the VPN, reply to Parag, Devin runs `plans/PLAN4.md`, skills
  files, synthetic weekend run, pilot.

## Where things live in the repository

- `llm_research/` (this folder): research, design reasoning, methodology, findings, references,
  decisions, correspondence, and the current proposal and ARCTIC request.
- `docs/`: documentation for the code base (design, desktop setup, runbook, weekend checklist,
  config reference, decision table).
- `plans/`: PLAN.md to PLAN4.md (PLAN4 is current) and the build log `PROGRESS.md`.
- `manualwork.md` (repo root): every manual step Carter has to do.

## Files

| File | Contents |
|---|---|
| `01_project_overview.md` | Research question, hypotheses, scope, novelty, timeline, repo |
| `02_decisions_log.md` | Every current decision, and a separate table of superseded ones |
| `03_pipeline_and_methods.md` | Weekly cycle, jobs J1 to J9, data, baseline, LLM ratings, views, Black-Litterman, strategies, guardrails, failure handling |
| `04_models_compute_arctic.md` | Where each step runs, desktop specs, all benchmark numbers, weekend run times, the ARCTIC request in detail, how the request shrank |
| `05_evaluation_methodology.md` | The current test plan per hypothesis, statistics, multiple testing, metrics, and what the code still needs |
| `06_evaluation_research_findings.md` | The research behind the evaluation plan: metrics primer, shuffle replacements, power, Sharpe conventions, Holm |
| `07_dropped_methods_and_why.md` | Shuffle test, small-model deployments, Slurm and CPU on ARCTIC, local open-weight models, other rejected choices |
| `08_literature_and_references.md` | The 51 proposal references by topic, evaluation sources, and the 10 removed after reading abstracts |
| `09_arctic_process_and_model_facts.md` | ARCTIC access routes, allocation caps and windows, document rules, gateway facts, both model cards, what is still unknown, sources |
| `13_litellm_gateway_reference.md` | LiteLLM docs link (https://docs.litellm.ai/docs/) and the user-side facts: keys, limits, errors, response headers, reasoning, fallbacks |
| `10_open_items_and_next_steps.md` | Short summary; Devin's work is `plans/PLAN4.md`, Carter's is `manualwork.md` |
| `11_correspondence.md` | Parag's feedback, the reply draft, Andalib status |
| `12_ai_hallucination_research.md` | The earlier hallucination research (dropped 2026-10-20) |
| `current_documents/` | Current `ARCTIC_Resource_Request` and `Project_Proposal_IEEE` (PDF and LaTeX). No older versions |

## How to keep this folder current

- When a decision changes, update `02_decisions_log.md` (move the old one to section B with the
  date) and fix every file that mentions it.
- Dropped methods go in `07_dropped_methods_and_why.md` with the reason, never left as current.
- Replace the files in `current_documents/` whenever the proposal or ARCTIC request changes.
- Estimates are labeled as estimates; replace them with measurements when available.

## Writing rules used throughout

No em dashes. "progsu" in lowercase. No invented facts. First person in ARCTIC documents (no
third-person self-reference). No lines ending on only 2 to 5 words in documents. The banned word for
experimental conditions is never used (`scripts/check_banned_words.py` enforces it).
