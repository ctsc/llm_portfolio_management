# 04 Models, compute and the ARCTIC request

Last updated: 2026-11-17.

## Where everything runs

| Work | Where | Why |
|---|---|---|
| LLM ratings, Gemma 4 31B and MiniMax M2.7 | ARCTIC LiteLLM gateway, `https://litellm.rs.gsu.edu/v1` (GSU network or VPN) | Far too large for a 6 GB GPU; ARCTIC already hosts both |
| LLM ratings, API models (Claude Haiku 5.5, GPT-6 Luna) | Provider batch APIs | Paid, cheap models with batch and prompt caching |
| Kronos baseline | Desktop GPU (RTX 3050) | Kronos-base is small (102M parameters) |
| Data snapshot, validation, Black-Litterman, all 5 strategies, controls, simulator, evaluation, backups | Desktop CPU | Measured to fit easily (below) |

## Carter's desktop (measured 2026-11-10)

| Part | Spec |
|---|---|
| Name | ctierney2, Windows |
| CPU | AMD Ryzen 5 8400F, 6 cores / 12 threads (not the Intel chip he first thought) |
| RAM | 31.6 GB |
| GPU | NVIDIA RTX 3050, 6 GB (about 1.2 GB used by everyday apps), driver 595.95, CUDA 13.2 |
| Disk | 359 GB free on C: |
| Linux | WSL2 Ubuntu (user `ctsc`), sees the GPU; Python 3.11.18 venv at `~/pm311` via uv; torch 2.15.0+cu132 |

Run the pipeline from `~/pm` inside Ubuntu, not from a OneDrive folder (see
`docs/desktop_setup.md` in the repository).

## Benchmarks on the desktop (2026-11-10)

`scripts/benchmark_pipeline.py --sources 3 --weeks 24` (synthetic covariance, solver time only).
Each strategy solved 4 times (3 sources + Q0).

| Strategy | 300 stocks, mean solve (s) | 100 stocks, mean solve (s) | Ratio |
|---|---|---|---|
| MVO | 0.079 | 0.009 | about 9x |
| Mean-CVaR | 114.4 | 21.6 | about 5x |
| Robust MVO | 3.60 | 0.020 | about 180x |
| Resampled MVO | 11.7 | 1.1 | about 11x |
| Top-K | 0.003 | 0.001 | |
| HRP | 0.011 | 0.004 | |
| Equal weight | 0.000 | 0.000 | |

| Measure | 300 stocks | 100 stocks |
|---|---|---|
| J7 seconds per week at 3 sources | 519 | 91 |
| J7 seconds per week at 51 sources (worst case) | 6,748 (112 min) | 1,181 (20 min) |
| Peak memory | 1.4 GB | 0.6 GB |

Kronos smoke test (`scripts/kronos_smoke.py`, 20 samples per stock, synthetic prices):

| Measure | 300 stocks | 100 stocks |
|---|---|---|
| Model load | 18.6 s (first run includes a 409 MB download) | 3.9 s |
| Forecast | 3,373 s (56 min) | 195 s |
| Peak GPU memory | 4.49 GB | 1.79 GB |
| Missing forecasts | 0 | 0 |

### Why 300 stocks took so much longer than 100

- Optimizers: the covariance matrix grows with the square of the stock count (300^2 is 9 times
  100^2), so MVO and Resampled grow about 9 to 11x. CVaR grows about 5x (10,000 scenarios times n
  weights). Robust grew about 180x, worse than cubic (27x); a likely but unverified cause is the
  dense matrix work no longer fitting in CPU cache.
- Kronos should scale about linearly (expected about 590 s at 300), but took 17x. Likely cause:
  4.49 GB plus about 1.2 GB used by other apps nearly filled the 6 GB card, and Windows spilled
  into slower shared system memory. Not confirmed. Fixes: close other GPU apps, or forecast in
  chunks of 100 stocks (expected about 10 minutes).
- Compared with Devin's build VM, the desktop was slower on CVaR (114 s vs 22 s) and Robust
  (3.6 s vs 0.41 s). Both still fit.

## Weekend run time on the desktop (300 stocks)

| Job | Time | Status |
|---|---|---|
| J1 snapshot | Minutes | Not yet measured |
| J2 Kronos | 10 to 56 min | Measured (56 min with other apps open) |
| J3 submit API batches | Minutes; providers allow up to 24 h to return | Provider limit |
| J4 gateway ratings, 4,200 requests | About 4 to 9 h | Estimate only (30 to 60 s per request, 4 at a time per model); MiniMax reasoning of about 2,000 tokens per answer could roughly double it, which also raises the VPN session-limit risk |
| J5 collect API batches | Usually done by the end of J4 | |
| J6 validate | Minutes | Not yet measured |
| J7 portfolios | About 63 min at 28 sources (estimated from the per-source slope); 112 min at 51 | Benchmark |
| J8 replay and score | Seconds (about 0.02 s per 24-week replay) | Measured earlier |
| J9 backup | Minutes | |
| Total | About 6 to 12 h, mostly waiting on J4 | Fits a Friday-night-to-Monday window easily |

At 100 stocks every local step is about 3 to 10 times faster. End of study: downloading French
factors and running every test takes seconds to minutes. Nothing at the end of the study needs
heavy compute now that the shuffle test is dropped.

## The ARCTIC allocation (granted 2026-11-16)

ARCTIC granted the default startup allocation: 1,000 GPU service units and 10,000 CPU service units,
measured as GPU and CPU usage (one GPU service unit is one V100 32 GB for one hour). It is less than
the 2,730 GPU-hours requested for 300 stocks and more than the 910 requested for 100.

| Option | Estimated use (24 weeks) | Fits in 1,000? |
|---|---|---|
| 100 stocks | about 607 GPU-hours | Yes, about 40% headroom |
| 300 stocks | about 1,820 GPU-hours | No |

Decision B4: 100 stocks. The estimates (40 GPU-seconds per Gemma request, 90 per MiniMax request)
are likely high (shared batching and prefix caching on the gateway), so the measured timing run
could show room for more; PLAN4 allows a change before the freeze only with a 25% margin. The
weekly run tracks estimated usage against the 1,000 limit (N7). The 10,000 CPU service units are
not needed: everything else runs on the desktop.

## The ARCTIC request (current)

File: `current_documents/ARCTIC_Resource_Request.pdf` (3 pages, first person), revised 2026-11-12
after Andalib's reply. Written as a standard (ARCA) allocation request with Parag as PI. Finalized
2026-11-16 for Parag to submit through Elpis as the supporting document (Andalib's instruction; no
Hydra ticket). Andalib said the usage numbers look good; how ARCTIC counts gateway use and each
key's limits are still being decided (`09_arctic_process_and_model_facts.md` section G). The
2026-11-16 edits: four skills prompts (not four to five), both API models named, logging of token
counts and gateway IDs mentioned, a start date (pilot ideally early December 2026, live run through
spring 2027), and a slightly narrower figure. If an allocation turns out to be
needed, the required Code Performance and Scaling document still has to be written from measured
gateway requests.

What it asks for:
- Gateway LLM calls on the two existing models only. No new model deployments, no gateway changes,
  no CPU hours, no storage beyond the default project space. The small models and Slurm were not
  re-added when Andalib offered (2026-11-12).
- Use pattern: one stock per request, about 300 to 8,000 input tokens, about 300 output tokens for
  Gemma 4 31B and up to about 2,000 for MiniMax M2.7 (its reasoning counts as output, confirmed by
  Andalib). Sent in a 48-hour weekend window, both models in parallel, up to 4 requests at a time
  per model, no images, tools or web access. About 4,200 requests per weekend at full scale:
  roughly 1.3 to 34 million input and 4.8 million output tokens.
- Six prompt types are planned; the request is sized for seven as headroom.
- Temperature 0 and a fixed seed; the PDF says the pilot measures how often repeated ratings agree
  and that a small fixed set of check prompts runs each weekend to catch model changes.
- Duration: 12 months, heavy use only in about 24 weekend windows.

Sizing (all per-request costs are estimates):

| Option | Requests per week | Tokens per week (in; out) | Weeks | Estimated use | Request |
|---|---|---|---|---|---|
| 300 stocks (preferred) | 4,200 (300 x 7 x 2) | 1.3 to 34M; 4.8M | 24 | 1,820 GPU-hours | 2,730 GPU-hours |
| 100 stocks (smaller) | 1,400 | 0.4 to 11M; 1.6M | 24 | 607 GPU-hours | 910 GPU-hours |

Basis: about 40 V100-equivalent GPU-seconds per Gemma request and about 90 per MiniMax request,
from a memory-bandwidth estimate (generating each output token reads the active weights once per
batch; MiniMax writes up to about 2,000 tokens). At 300 stocks each model gets 2,100 requests a
week: about 560 GPU-hours for Gemma and 1,260 for MiniMax over 24 weeks, 1,820 in total. The
request adds about 50% for failed calls, retries, test and check calls (the pilot repeat check is
about 1,200 requests, canaries about 20 per weekend), and a longer live run. Working in
`llm_research/correspondence/andalib_reply_2026-11-12.md` section E (gitignored).

History of this sizing: the first version (2026-11-10) used 40 GPU-seconds for every request and
300 output tokens, giving 1,120 estimated and 1,700 requested at 300 stocks (373 and 600 at 100).
Andalib's reply showed MiniMax's reasoning must be counted as output, which raised the estimate.

Other statements in the request: a gateway outage leaves that model's week out of its statistics;
API models run on provider servers with no ARCTIC compute; all inputs are public market data and
news with no personal or regulated data; ARCTIC's hosted models are approved for unregulated
university data; models banned by GSU's registry are not used; papers will disclose AI use.

Still to correct after measurement: the 40 and 90 GPU-second estimates, from the timing run
(`--time-gateway`) and ARCTIC's metering answer.

## How the compute request shrank (history, for explaining it to others)

1. The first committed request asked for about 15,000 CPU hours (the earlier projection was about
   10,968 CPU hours), mostly for the shuffle test across all strategies and portfolios.
2. Narrowing the shuffle to MVO, Robust and Top-K (CVaR and Resampled were about 99% of the
   shuffle compute) and using measured solve times instead of estimates cut it to a 1,500 CPU-hour
   request.
3. Moving every non-LLM step to the desktop (2026-11-10) removed CPU hours from the request
   entirely.
4. Dropping the shuffle test (2026-11-11) removed the last heavy compute job, even locally.

## Risks of running on one desktop

- A Windows Update restart, sleep, a power blip or a VPN drop mid-weekend can cost a week.
  `run_weekend.py` resumes where it stopped, which helps.
- Before going live: set sleep to Never; pause Windows Update for the study (active hours max out at
  18 hours a day, so they cannot cover a weekend; renew the pause before it ends); keep the GSU VPN
  connected during J4; make sure J9 copies the backup off the desktop. PLAN4 Phase 3A adds
  keep-awake, VPN outage retries, per-job checkpoints, progress and a weekly study log.
- Close GPU-heavy apps (games, Overwolf, Discord, Teams, Steam) during J2.
