# Manual work (Carter only)

Every manual step still open across the three plans, in the order to do them. Moved here from
`PLAN3.md` Phase 7, `PLAN2.md` Phase 15 and `PLAN.md` Phase 17 (the v1 items were already folded
into Phase 15; the crosswalk at the end shows where each went). The letters and numbers below
keep the `PLAN2.md` Phase 15 labels, because `README.md`, `PROGRESS.md` and `docs/` point at them
as "PLAN2 phase 15 B" and so on. Nothing here is done by Devin: it needs accounts, keys, money,
the ARCTIC allocation, research content or a judgment call.

Already done and removed from this list: PLAN2 phase 15 A1 (repository access and Carter-only
commits), A2 for the v2 and v3 builds (every commit is `ctsc`), D1 (the repository is on GitHub
with CI green; GitLab is the fallback mirror and is out of free minutes), the
`docs/desktop_checklist.md` half of D4 (replaced by `docs/weekend_checklist.md` in PLAN3), and
PLAN3 phase 7 item 4 (no change to the ARCTIC request is needed).

## 0. From PLAN3 Phase 7 (do first)

1. Ask Andalib (ARCTIC) to confirm the real maximum job length and the memory-per-core policy.
   If the limit differs from 24 hours, set `MAX_JOB_HOURS` in `scripts/arctic/arctic.env` to stay
   well under it (`common.sh` refuses any walltime above it; W1).
2. Delete the untracked `docs/arctic_gpu_request.md` (also settles PLAN2 phase 15 D4) and the
   untracked duplicate `Project_Proposal_IEEE.pdf` at the repository root (the committed copy is
   `docs/proposals/Project_Proposal_IEEE.pdf`).
3. Read the new `README.md` top to bottom against `docs/proposals/Project_Proposal_IEEE.pdf`.
4. Then continue with PLAN2 phase 15 below.

## PLAN2 Phase 15

### A. Review of the unattended builds
2. Skim `git log` (every commit should show only `ctsc`) and the `PROGRESS.md` entries for v2
   phases 1 to 14, the review fixes and v3 phases 1 to 6, including each "Choices made while
   unattended" section.
3. Decide on every decision a phase flagged: the shuffle scope and compute projection (now G6:
   MVO, Robust MVO and Top-K only, 85 CPU hours at 300 stocks and 5.6 at 100 over 24 weeks),
   universe size (B4, `config/study.yaml` keeps 100 until the allocation is known), and any
   `view_shift` warning from the benchmark.
4. Rollback, if ever needed: v1 is commit `030faf4`; revert any phase with git.

### B. Accounts, keys, and spending limits
1. Create API accounts and keys for Anthropic (Claude Haiku 5.5) and OpenAI (GPT-6 Luna). Put
   the keys in the `.env` file on ARCTIC only (names in `docs/config_reference.md`). Never commit it.
2. Set a monthly spending cap in each provider dashboard, so a bug cannot run far past the D3
   budget.
3. Create free accounts and keys for Alpaca (prices and news), Finnhub (earnings dates) and FRED
   (T-bill rate). Confirm Alpaca's free plan includes the news endpoint and full daily bars; if
   not, switch `data.price_provider` to yfinance in config.
4. Download the official Fama-French five-factor daily file from the Ken French Data Library when
   it is needed for the final H5 run, and set `eval.fama_french_file` in config.

### C. ARCTIC allocation and environment
1. Submit the ARCA proposal through Dr. Tamhankar. Once granted, fill `scripts/arctic/arctic.env`
   from `arctic.env.example`: account, partitions, `GPU_GRES`, `GPU_MEM`, every walltime (each at
   most `MAX_JOB_HOURS`), `POLL_INTERVAL_MINUTES`, storage path, module and venv.
2. Set up the Python 3.11 venv on ARCTIC, clone the repository there, and run the test suite once
   on a login node.
3. Confirm GSU network or VPN access to `https://litellm.rs.gsu.edu/v1`, create the gateway API
   key, and store it as `ARCTIC_LITELLM_API_KEY` in `$HOME/.env` on ARCTIC only (mode 600).
4. Ask ARCTIC whether compute nodes reach the internet and the gateway, and set
   `ARCTIC_NETWORK_MODE` (`compute` or `login`). Ask about inode and file-count quotas.
5. When ARCTIC deploys Gemma 4 E4B, Phi-4 Mini and Llama 3.2 3B, enter their gateway model IDs in
   `config/models.yaml` and enable them.
6. Clone Kronos at the pinned commit into `vendor/kronos` (set `kronos.repo_commit`) and download
   `Kronos-base` and `Kronos-Tokenizer-base` from the `NeoQuasar` Hugging Face organization into
   `HF_HOME`.
7. Accept the Hugging Face licenses for Gemma and Llama, and download the fallback model weights
   into `HF_HOME`.

### D. Data folders and backups
2. Delete the old synthetic `data/` folders before the first v2 run (v1 and v2 file formats
   differ; nothing real is lost).
3. Run `run_weekend.py --synthetic` once on ARCTIC, download the backup zip it prints, then run
   one restore by hand to confirm it works (`docs/runbook.md`).

### E. Smoke tests and benchmarks on ARCTIC
1. Run `scripts/smoke_real_services.py` for every service (one real call each, printing status,
   cost and latency). Fix every UNVERIFIED item it exposes (list below and per phase in
   `PROGRESS.md`): the data adapters (Alpaca, yfinance, Finnhub, FRED, Wikipedia), the Anthropic
   and OpenAI batch providers, the Kronos vendored API, and the v2 items (gateway JSON mode, seed
   support, MiniMax reasoning output and output budget, Kronos per-path samples, fallback server
   choice).
2. Run `scripts/benchmark_pipeline.py --n-stocks <size> --sources 3 --weeks 24` on ARCTIC:
   gateway seconds per call and safe concurrency per model, Kronos GPU time over the full
   universe, J7 time, and one shuffle replay per G6 strategy (pass it as `--replay-seconds`).
   Confirm the whole weekend fits between the Friday night or Saturday morning start and Monday
   09:00 ET with margin, and that every job stays under 12 hours (W1). Use the results to set B4
   (100 to 300), `llm.concurrency` and `eval.shuffle.shard_size`, and compare the projected shuffle
   CPU hours with the allocation.
3. Start `llm_fallback.sbatch` once and run one small model through it end to end with
   `--gateway-fallback`.

### F. Reviews
1. Read the base instruction wording in `prompts/` and approve or rewrite it, then run
   `scripts/refresh_manifest.py`.
2. Read the report header from the synthetic run and confirm the design choice values it prints.
3. Check the `view_shift` warnings from the benchmark (PLAN2 section 8).

### G. Research content only Carter writes
1. Write the 4 to 5 skills files and the placebo text in `prompts/`. Run
   `scripts/check_placebo_length.py`, then `scripts/refresh_manifest.py` to freeze versions and
   hashes.
2. Check the real skills file token counts against the 6000 cap (C4), and confirm or change C4
   and the 150-token rationale cap (C7).
3. Build the stock list (B2): run `build_universe` at the chosen size, review the printed sector
   table and the thin-sector log, add any `extra_tickers` by hand, then freeze the universe file
   with a version tag.
4. Enter current API model prices in `config/models.yaml` (the committed values are placeholders).
   Test-run the real prompts on a few stocks for each model, then run `scripts/estimate_cost.py`
   and recompute the budget (D3). Decide whether any model needs a `max_output_tokens` override
   (none is set now).
5. Set `timing.live_end_date` (A3).

### H. Freeze, pilot, and go
1. Run a dry run on the last 3 to 4 real weeks with real providers, only as a bug check. Do not
   tune prompts or settings on the returns.
2. Tag the repository `frozen-v2`. From then on any change is a new version, documented and
   flagged in the analysis.
3. Run the one-week pilot with `scripts/arctic/submit_weekend.sh`. Review job success,
   valid-output rates by model and prompt type, gateway errors, fallback use, batch return times,
   finish-job resubmissions, cache hits, cost and run time. Hand-check trades, costs and stops
   against the simulator output.
4. Fix engineering problems only, record each fix, and drop or swap any model that failed
   reliability or cost limits (with a log entry saying why).
5. Start the live run the week after the pilot passes (A2). The pilot week stays excluded from
   all results.
6. Every weekend: run `submit_weekend.sh` Friday night if possible (Saturday morning as the
   backup), read the finish job's summary, and pull the dated backup (`docs/weekend_checklist.md`).
7. After the last live week: `portstudy shuffle plan`, submit `shuffle_array.sbatch`,
   `portstudy shuffle collect`, then `portstudy eval`.

## UNVERIFIED items (closed by E1 and E2)

- Every ARCTIC value in `scripts/arctic/arctic.env.example` (account, partitions, GRES, GPU
  memory, walltimes, poll interval, module, venv, data root, `HF_HOME`, network mode, array
  throttle), and the real walltime limit behind `MAX_JOB_HOURS`.
- The serving command in `scripts/arctic/llm_fallback.sbatch` (vLLM assumed) and the
  fallback-eligible model ids.
- Gateway behaviour: JSON mode, seeds, rate limits, MiniMax reasoning output and whether it needs
  an output budget, small-model ids.
- Kronos per-path sampling (`predict_paths`) on the real checkpoint; the 20-call fallback costs
  about 20x the GPU time.
- ARCTIC core speed relative to the build VM, so the 85 CPU hour shuffle projection at 300 stocks
  (5.6 at 100) is a VM figure until E2.
- D3, the paid API budget, after the G4 test prompts.
- Slurm behaviour of the three-job layout and the shuffle array on ARCTIC (dry-run only so far).

## Crosswalk: PLAN.md Phase 17 to the items above

| PLAN.md 17 item | Here | Change from v1 |
|---|---|---|
| A1 API accounts and keys | B1 | keys live on ARCTIC |
| A2 spending caps | B2 | none |
| A3 Alpaca, Finnhub, FRED accounts | B3 | none |
| A4 Fama-French file | B4 | none |
| B1 drivers, CUDA, Python, Ollama | C2 | replaced: ARCTIC venv, no local GPU, no Ollama |
| B2 pull local models | C5 | replaced: gateway models; Qwen removed (GSU tool registry) |
| B3 Kronos clone and weights | C6 | on ARCTIC |
| B4 smoke tests | E1 | adds the v2 UNVERIFIED items |
| B5 benchmark | E2 | on ARCTIC; the 6 GB swap rule no longer applies |
| B6 local power settings | none | dropped: nothing runs on a local machine |
| C1 GitLab project and CI | done | GitHub is the primary host, GitLab the fallback |
| C2 synthetic run and restore | D3 | on ARCTIC |
| D1 skills file and placebo | G1 | 4 to 5 skills files, placebo matched to their mean |
| D2 token caps | G2 | none |
| D3 stock list | G3 | size chosen after the allocation, thin-sector log added |
| D4 prices and budget | G4 | gateway models cost $0 |
| D5 live end date | G5 | none |
| E1 to E6 freeze, pilot, go | H1 to H6 | tag `frozen-v2`, weekend runs through `submit_weekend.sh` |
