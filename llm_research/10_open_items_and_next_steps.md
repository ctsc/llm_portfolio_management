# 10 Open items and next steps

Last updated: 2026-11-17.

Remaining work lives in two files at the repository root:

- **`plans/PLAN4.md` Phases 1 to 11 (Devin; build status in `plans/PROGRESS.md`, Phases 1 and 2
  done 2026-11-17):** remove the shuffle test and the ARCTIC Slurm layer and small models;
  desktop runtime (Kronos chunking, live-readiness checks, lock file, gateway timing, backup check);
  decisions, evaluation config and `config/preregistration.yaml`; statistics primitives and metrics;
  French factors and stock characteristics; signal-level and portfolio-level hypothesis tests; the
  report; a docs recheck; end-to-end verification.
- **`manualwork.md` (Carter only):** the single list of manual work, replacing every manual
  phase in PLAN.md to PLAN4.md.

A full code and plan review on 2026-11-11 is in `plans/REVIEW5_fable_2026-11-11.md`. Carter approved
three method changes (neutral LLM view E7, strategy sensitivity calibration S8, stop reset G1b) and
PLAN4 now includes them and the review's bug fixes (new Phase 3B, larger Phases 3 and 3A). On
2026-11-12 Carter settled the rest: four pooled main comparisons (G7), a fixed view scale of 0.02
(A8), the Friday 16:00 ET cutoff unchanged (A9), the pilot excluded with a clean start (A7), a
five-day no-price liquidation (U2), the `momentum_only` Kronos fallback (K3), confidence-only
control portfolios for Robust and Resampled MVO (G19), and four skills files (B1). No review
decision is open. Andalib answered on 2026-11-12: temperature 0 is confirmed and the request is
revised. The request was submitted through Elpis and approved (LiteLLM access granted, recorded
2026-11-17). ARCTIC granted the default startup allocation (1,000 GPU and 10,000 CPU
service units, metered as GPU and CPU usage), so the study uses 100 stocks. Still to confirm: the
key's limits.

The most time-sensitive items in `manualwork.md`:

1. Email Andalib to be added to the project's team on the gateway (no "Create New Key" button
   yet), then create the two keys (`portstudy-weekend`, `portstudy-dev`).
2. Email the GSU IT help desk (help@gsu.edu, 404-413-4357) about VPN session and idle limits; every
   gateway request needs the GSU network or VPN.
3. Hand `plans/PLAN4.md` to Devin (Phase 1 first). Devin builds as if every key exists.
4. Once the key exists: run the gateway timing tool and replace the per-request GPU-second
   estimates (N7).
5. Before the pilot: write the four skills files and the placebo, build the 100-stock list, set the
   view-sensitivity threshold and freeze `config/preregistration.yaml` (the main comparisons and
   residual-IC controls are already decided, G7 and G9).

Also decided on 2026-11-17: main comparison 1 is the LLM-minus-baseline rank IC gap (G7), every
metric is saved for every unit in `all_metrics.csv` with Sharpe net of the T-bill rate leading the
portfolio table (G22), and both API models are used (D1). The proposal was updated to match.

Repository housekeeping done on 2026-11-11 (not yet committed; PLAN4 Phase 1 commits it): plans
and the build log moved to `plans/`; this folder moved into the repository; `docs/` rewritten to the
current design with "pending PLAN4" notes; `docs/desktop_setup.md` added; duplicate proposals and
stale ARCTIC docs removed.
