# Manual work (Carter only)

Last updated: 2026-11-17 (after the full review in `plans/REVIEW5_fable_2026-11-11.md`, Andalib's
replies, the ARCTIC approval and the final PLAN4 review). This is the only list of work Carter has to do by hand: accounts, keys,
money, research content, judgment calls, the desktop, and emails. Nothing here is done by Devin.
It replaces the manual phases of `plans/PLAN.md`, `PLAN2.md`, `PLAN3.md` and `PLAN4.md`. Devin's
build work is in `plans/PLAN4.md` Phases 1 to 11; do the items below after Phase 11 unless an item
says otherwise (sections 0 and A can start now).

Dropped from the older lists because they no longer apply: ARCTIC Slurm setup, `arctic.env`,
walltime and memory limits, the vLLM fallback server, small-model IDs, the shuffle array, Ollama,
and the GitLab items already done.

## 0. Start the build
1. Done 2026-11-16: the repository moved to `C:\dev\portfoliomanagement`; the old OneDrive copy is
   renamed `portfoliomanagement_olddd`. Delete the old copy only after PLAN4 Phase 1 Step 0 passes.
   The correspondence files are gitignored and no longer backed up by OneDrive; keep a copy
   elsewhere.
2. Hand `plans/PLAN4.md` to Devin, starting at Phase 1. Devin commits and pushes as `ctsc` only,
   with no AI or co-author lines; check `git log` after each phase. Confirm
   `llm_research/11_correspondence.md` never appears in a commit (it is gitignored).
3. Done 2026-11-17: deleted the empty `vendor/` folder left by the failed Kronos clone.

## A. Communication and the ARCTIC request
1. Done 2026-11-16: Parag submitted the request in Elpis and ARCTIC approved it (item 7). Send
   Parag the updated `Project_Proposal_IEEE.pdf` (2026-11-17: four skills files, 100 stocks, both
   API models, Sharpe) if he wants the current version.
2. Done 2026-11-11: emailed Andalib the seven questions. He answered on 2026-11-12 (organized in
   `llm_research/correspondence/andalib_reply_2026-11-12.md`; facts in
   `llm_research/09_arctic_process_and_model_facts.md` section G). Since then access was granted through
   Elpis (item 7); how gateway requests convert into GPU hours and each key's limits are still open.
3. Done 2026-11-12: no Hydra ticket for the request itself; access went through Parag's Elpis request.
   The wiki's "Get access" step for joining the project's team on the gateway does use a Hydra
   ticket, so with Hydra still failing, ask Andalib instead (item 6). Resetting your GSU password
   may fix Hydra, since ARCTIC services share it.
4. Done 2026-11-12: replied to Andalib with the revised request. He said the usage numbers look
   good, confirmed MiniMax's reasoning arrives in `reasoning_content` with `reasoning_tokens` in
   `usage`, and listed what can be logged to catch model changes (now in PLAN4 G21).
4a. Done: Parag submitted the request in Elpis, and ARCTIC approved it (LiteLLM access granted,
   recorded 2026-11-17). Was: send Parag the final `llm_research/current_documents/ARCTIC_Resource_Request.pdf`
   (finalized 2026-11-16) and the Elpis steps and abstract text in
   `llm_research/correspondence/elpis_submission_2026-11-16.md`. Parag submits it in Elpis
   (https://elpis.rs.gsu.edu): his project, Request Resource Allocation, the abstract, add you
   (`ctierney2`) as a user, attach the PDF as the supporting document.
5. Now: email the GSU IT help desk (help@gsu.edu, 404-413-4357) about VPN session and idle limits.
   Every gateway request needs the GSU network or VPN (wiki, rechecked 2026-11-17), and a weekend
   run can take 4 to 9 hours. Record the answer in `llm_research/09_arctic_process_and_model_facts.md`.
   Either way, watch for drops in the pilot; the run resumes from the last saved unit.
5a. If ARCTIC asks for the Code Performance and Scaling document (the policy page lists it as
   required for allocations; Andalib asked only for the PDF), write it from the measured gateway
   numbers in `data/outputs/gateway_timing.csv` once you have a key.
5b. When ARCTIC says how use is metered, and after the timing run, correct the request's per-request
   estimates (40 GPU-seconds per Gemma request, 90 per MiniMax request) and resend it if the total
   changes.
6. Get the LiteLLM key (approved 2026-11-16; Andalib added you to the project as a Manager). Steps
   from the wiki's "Get access" section (full detail in
   `llm_research/09_arctic_process_and_model_facts.md` section G):
   - On the GSU network or VPN, sign in once at https://litellm.rs.gsu.edu/ui with your campus ID.
   - Status 2026-11-17: signed in, but the Virtual Keys page shows no "Create New Key" button, so
     you are not on the project's team yet; email Andalib to add you to the team (the email also
     asks whether a user key can read `/v1/model/info` and who sets per-key limits).
   - When they confirm, create keys at https://litellm.rs.gsu.edu/ui, Virtual Keys, "+ Create New
     Key", Team = the project. Make two: `portstudy-weekend` (the weekly run) and `portstudy-dev`
     (timing runs and tests), so either can be replaced alone. Each is shown once: save it in your
     password manager.
   - Store the weekend key as `ARCTIC_LITELLM_API_KEY` in `~/.env` inside WSL (mode 600, never
     committed, never pasted into chats or files), and check access with
     `curl -sS https://litellm.rs.gsu.edu/health/liveliness` on the VPN.
   - Ask Andalib what limits the key has (rate, concurrency, tokens), if the key page does not say.
7. Done 2026-11-16: ARCTIC granted the default startup allocation (1,000 GPU and 10,000 CPU service
   units), so the study uses 100 stocks (B4); `universe.size` is already 100. After the timing run,
   replace `arctic.gpu_seconds_per_request` with the measured values (N7). Every month during the
   study, check the real GPU and CPU usage in Elpis and note it in the study log; startup
   allocations cannot be renewed, so more GPU-hours would need a research allocation request (next
   window January 15 to February 1, starting April 1).
8. Add citation numbers to the proposal's Related Work paragraph (optional, documents only).

## B. Desktop setup (see `docs/desktop_setup.md`)
1. Re-create the WSL checkout with `git clone` into `~/pm` (replace the copied folder from
   2026-11-10), create the Python 3.11 venv, install from `uv.lock` with the `ml`, `solvers`,
   `data`, `api` and `stats` extras.
2. Pick the Kronos commit (the latest commit on the day you set it up; record it and never change
   it after the freeze), clone it into `vendor/kronos`, set `baseline.kronos.repo_commit`, set
   `baseline.forecaster: kronos`.
3. Run `python scripts/kronos_smoke.py --n-stocks 100 --batch-size 100` with other GPU apps closed;
   confirm peak GPU memory fits and record the time.
4. Set `backup.rsync_target` to a folder WSL can write (check with the J9 probe) and decide how the
   backup reaches cloud storage.
5. Apply the reliability settings: sleep Never during runs; pause Windows Update (Settings, Windows
   Update, Pause updates, up to 5 weeks at a time) and renew the pause before it runs out during the
   study. Active hours cannot cover a weekend (they max out at 18 hours a day).
6. Confirm on the desktop that the keep-awake helper and the WSL clock check from PLAN4 Phase 3A
   work (run a synthetic weekend, check `powercfg /requests` in PowerShell while it runs).

## C. Accounts, keys and budget
1. Anthropic and OpenAI API keys with monthly spending caps; free keys for Alpaca, Finnhub and FRED
   (confirm Alpaca's free plan includes news and full daily bars from the consolidated SIP feed for
   the study dates; the code used to request the single-exchange IEX feed. If SIP is refused,
   set `data.price_provider: yfinance`). Keys go in `~/.env` only.
2. Enter real API prices in `config/models.yaml`, test the real prompts on a few stocks per model,
   run `scripts/estimate_cost.py`, and set the budget (D3). Keep both API models (Claude Haiku 5.5
   and GPT-6 Luna) if the estimate fits the 50 USD goal; if not, drop the more expensive one and
   record it in PROGRESS.
3. Decided 2026-11-12: temperature 0 for every model (D4). Both model cards recommend 1.0; the paper
   says why 0 was used. Andalib confirmed the gateway passes `seed` through but outputs are not
   guaranteed identical because requests are batched with other users'; the pilot repeat check
   (F2) measures how often they agree.
4. After the gateway timing run, set each gateway model's `max_output_tokens`. Starting values:
   Gemma 300 and MiniMax 2,000 (MiniMax always thinks, the thinking counts toward the limit, and
   ARCTIC suggests about 2,000). Raise MiniMax's limit if `finish_reason: length` appears; more than
   15% invalid answers sends its weeks to the Q0 fallback. Set each model's
   `request_timeout_seconds` from the same run (MiniMax starts at 600).
5. Enter a dated `snapshot_id` for each enabled API model in `config/models.yaml` (D5). The
   preflight refuses a live run while one is null, because an alias can move to a new model version
   during the study.

## D. Research content

Decided on 2026-11-11 after the review, already in PLAN4: the LLM view is centered on the Q0
forecast (E7), Mean-CVaR and Resampled MVO are rescaled on synthetic data before the freeze (S8),
and a stop on a rebalance day resets the stock's reference price (G1b).

Decided on 2026-11-12, also in PLAN4 (nothing left to choose here):
- **Main comparisons (G7, revised 2026-11-17):** exactly four pooled tests under Holm:
  LLM-minus-baseline rank IC gap, steered-minus-Q0 tilt IC, skills minus bare, skills minus
  placebo. Everything else is secondary, and every metric is saved for every unit (G22).
- **View scale (A8):** fixed at 0.02 for the LLM and baseline views every week; the trailing-IC
  switch is off (`views.ic_mode: fixed`).
- **News cutoff (A9):** Friday 16:00 ET, unchanged.
- **Pilot week (A7):** does not count; every portfolio starts from cash at the first live week
  (`views.ic_history_start: live_start_week`).
- **No-price liquidation (U2):** 5 trading days.
- **Kronos failure (K3):** `momentum_only`.
- **Confidence-only control portfolios (G19):** built for Robust and Resampled MVO, not Mean-CVaR.
- **Prompt types (B1):** four skills files, bare and placebo (six types); `skills_5` stays off.
- **Proposal wording.** If the proposal or paper states the view formula, the news timing or the
  CVaR setting, update it to E7, S8 and whatever you decide above.
0. Review the hypotheses I drafted at the end of the proposal (`llm_research/current_documents/`)
   and rewrite each one to what you actually expect, before the preregistration is frozen.
1. Write the 4 skills files (`skills_1` to `skills_4`; `skills_5` stays disabled) and the placebo
   in `prompts/`; run
   `scripts/check_placebo_length.py` and `scripts/refresh_manifest.py`; confirm the 6,000-token
   skills cap (C4) and 150-token rationale cap (C7).
2. Read and approve the base instructions in `prompts/base_instructions.md`.
3. Build and freeze the stock list (`build_universe` at the approved size; review the sector table
   and thin-sector log; add `extra_tickers` by hand if needed).
4. Fill `config/preregistration.yaml`: confirm the four pooled main comparisons Devin writes in
   (G7), keep the residual-IC control list as written (G9, it matches the proposal), and set the
   view-sensitivity threshold
   `view_sensitivity_min_tracking_error` (G13, from the Phase 3B calibration summary and the
   Phase 11 numbers); set `frozen: true` with the date. This must happen before the pilot. Then
   push the frozen file under a tag to the public repository so the date can be checked.
4a. Read `data/outputs/calibration/view_sensitivity.csv` and its summary (PLAN4 Phase 3B) and
   confirm or override the chosen Mean-CVaR `lam` and Resampled `draw_scale` before the freeze.
5. Set `timing.live_end_date` (A3) and `timing.pilot_week`.

## E. Real-service checks and dry run
0. Before the freeze, replace the synthetic canary set in `config/canary/` with 10 real tickers and
   their inputs from one past week (PLAN4 Phase 3A, G21). Never change it after the freeze.
1. `python scripts/smoke_real_services.py` for every service; fix any UNVERIFIED item it exposes
   (data adapters, batch providers, Kronos vendored API and per-path sampling, gateway JSON mode and
   seed).
2. `python scripts/smoke_real_services.py --time-gateway 20` per gateway model; replace the 4 to 9
   hour J4 estimate in README with the measured figure and set `llm.concurrency` if ARCTIC allows
   more or fewer than 4. Read the tokens in, tokens out (including MiniMax reasoning), seconds and
   `finish_reason: length` share it prints, then use them to correct the ARCTIC request (token
   ranges, GPU-seconds per request) and `llm_research/04_models_compute_arctic.md`. Check the
   Alpaca line reports the `sip` feed.
3. `python scripts/fetch_french_factors.py` once to confirm the URLs and file format.
4. `python scripts/run_weekend.py --synthetic` once on the desktop, download the backup, and restore
   it by hand once.
5. Dry run on the last 3 to 4 real weeks with real providers, as a bug check only (no tuning on
   returns). Run it with `PORTSTUDY_DATA_DIR` set to a scratch folder, never the live `data/`
   folder: dry-run weeks in the live folder would feed the trailing IC and every portfolio's
   starting positions (A7). The preflight refuses a live run if it finds a pre-pilot week.
6. `python scripts/check_live_ready.py` passes.

## F. Freeze, pilot, live
0. Before the freeze: start from an empty live `data/` folder and run `portstudy ledger rebuild`
   so the price ledger holds only rows from the configured feed; set
   `llm.gateway.features_frozen: true` once the timing run shows which features the gateway accepts.
1. Tag the frozen version (`frozen-v4`), push the tag. From then on any change is a new version,
   logged in PROGRESS and flagged in the analysis.
2. Run the one-week pilot with `python scripts/run_weekend.py`, then
   `portstudy repeat-check --week <pilot week>` (G20: 600 extra requests). Read
   `data/outputs/pilot_repeatability.csv` and keep the agreement numbers for the paper; the canary
   baseline is saved at the same time. Review job success, valid-output
   rates per model and prompt, gateway errors, fallback use, batch return times, cost and run time;
   hand-check trades, costs and stops; read each strategy's view shift (G13), the large-move flags
   and any removals; read each strategy's expected tracking error against Q0 (G13) and check that
   a stop on a rebalance day did not cause a Tuesday sale; replace the README's estimated J4 time
   with the measured one.
3. Fix engineering problems only; drop or swap a model that failed reliability or cost limits, with
   a PROGRESS entry.
4. Start the live run the week after the pilot passes (A2); the pilot week stays out of results.
5. Every weekend: Friday night (Saturday morning backup) run, read the summary and the new row in
   `data/outputs/study_log.csv` (including any `model_change` flag from the canary check; a flag
   does not stop the run, but note it), read `data/outputs/<week>/run_summary.md` and its "needs
   attention" list, skim `fallbacks.csv`, and confirm the `portstudy audit` result passed, check the backup, follow `docs/weekend_checklist.md`. Renew the
   Windows Update pause when it nears its end.
6. After the last live week: wait for the following Monday open, then run `portstudy closeout`
   (it refuses to run earlier). Then `python scripts/fetch_french_factors.py` (once the library covers the
   study window), set `eval.factor_source: french`, run `portstudy eval`, and write the paper from
   the report.
