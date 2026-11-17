# 07 Dropped methods and why

Last updated: 2026-11-11. Everything here is NOT part of the current study. It is kept so the
reasoning is not lost and old ideas are not reintroduced by mistake.

## 1. The shuffled-ratings test (H5). Dropped 2026-11-11

### What it was

For each portfolio, every week the LLM's ratings (low, likely, high, width) were randomly
reassigned across stocks, then the whole pipeline was rebuilt from those shuffled ratings (views,
Black-Litterman, the strategy's optimizer) and replayed over all weeks. View strength (the stored
weekly IC and view on/off flags) was held fixed, so only which stock got which rating was random.
Repeated 500 times, it gave 500 "fake LLM" portfolios. The p-value was (1 + number of shuffles at
or above the real result) / (500 + 1), on total return and Sharpe, with Holm within each
strategy. For Q0, the Kronos scores were shuffled instead. Code: `src/portstudy/eval/shuffle.py`.

### Why it was dropped

1. **It could never reach significance.** Holm ran within each strategy over 52 portfolios (51
   sources + Q0). With 500 shuffles the smallest possible p is 1/501, and Holm multiplies the
   smallest p by the family size: 52 x (1/501) = 0.104, which is above 0.05. Even an LLM that beat
   every shuffle could not pass. Passing would have needed more than about 1,040 shuffles or a much
   smaller family.
2. **It was the only expensive part of the evaluation.** It needed about 156 portfolios x 500
   shuffles x 24 weeks, about 1.9 million optimizer solves. On the desktop at 300 stocks that was
   638 CPU-hours (Robust MVO alone was 624, since one robust solve takes about 3.6 s there), about
   160 hours (about 7 days) with 4 processes in parallel. At 100 stocks it was 5.3 CPU-hours.
   Carter does not want any method that adds CPU or GPU hours.
3. **It was redundant.** Under random views, the expected active return vs Q0 is about zero before
   costs and negative after costs, so "beat Q0" is already at least as strict as "beat the middle
   of the shuffle distribution". The placebo prompt gives an empirical noise-view control for free,
   and rank IC, residual IC and factor regressions answer "skill or luck" in milliseconds.

### What replaced it

Residual IC (Fama-MacBeth), factor-neutral quantile spreads, daily Fama-French 5 plus momentum
regressions of the active return, the placebo control, and the deflated Sharpe ratio and t > 3
flag for any headlined portfolio. Details in `05_evaluation_methodology.md` and
`06_evaluation_research_findings.md`.

### Cheaper shuffle variants considered on 2026-11-10 (not adopted)

| Option | CPU-hours at 300 stocks | Could it reach significance? |
|---|---|---|
| As configured (MVO, Robust, Top-K; 500 shuffles; 52 per family) | 638 | No |
| A. Drop Robust, 2,000 shuffles for MVO and Top-K | about 57 | Yes (52/2001 = 0.026) |
| B. Shuffle only pre-named main portfolios (about 3 per strategy), 2,000 shuffles | about 150 | Yes (3/2001, about 0.0015) |
| C. Shuffle ratings only, no optimizer (IC level) | seconds | Yes, but mostly repeats H1 |

Carter chose to drop shuffling altogether and use the standard tests instead.

### Earlier history of the shuffle design

- Oct 2026: shuffle the best portfolio per model plus Q0; then 500 shuffles across all 13
  portfolios of the original single-optimizer design.
- Nov 1: 500 shuffles on all portfolios across the five strategies.
- Nov 8 (decision G6): only MVO, Robust and Top-K plus their Q0s (Mean-CVaR and Resampled were
  about 99% of the compute), as a Slurm array on ARCTIC CPUs with a local fallback.
- Nov 10: moved to the desktop with everything else; found the Holm flaw.
- Nov 11: dropped.

## 2. Requesting new small models on ARCTIC. Dropped 2026-11-10

Was: ask ARCTIC to deploy Gemma 4 E4B, Phi-4 Mini (3.8B) and Llama 3.2 3B on its gateway, with
Carter's own Slurm GPU jobs as a fallback. Why dropped: Parag asked to confirm model availability
and keep scope manageable; asking for new deployments risked denial when two capable models already
exist on the gateway. Now only Gemma 4 31B and MiniMax M2.7.

## 3. CPU hours, storage and Slurm jobs on ARCTIC. Dropped 2026-11-10

Was: weekend jobs (data, Kronos, LLM, finish job) and the shuffle array as Slurm jobs on ARCTIC,
with walltime caps (W1: at most 15 h, target 12; ARCTIC was said to cancel jobs over 24 h),
memory rules (M1: 3 GB per core), a submit script on ARCTIC (N2), a vLLM fallback server for
gateway outages (D2), and requests for CPU hours (15,000, later 1,500), development GPU/CPU and
disk. Why dropped: everything except the LLMs fits on Carter's desktop (benchmarked 2026-11-10),
so the request became GPU hours for gateway calls only. A repo review had also found several
Slurm-only bugs (sbatch scripts could not find `common.sh` under Slurm's spool folder, the LLM job
could start before the snapshot existed, the finish job had no safety net, the fallback server
wrote one URL for three models, the shuffle array had no account or walltime flags). Those no
longer matter unless Slurm is ever used again.

## 4. Running open-weight models on the local GPU. Dropped late Oct 2026

Was (Oct 25): Qwen 3.5 4B and Phi-4 Mini on the RTX 3050, Llama 3.2 3B as a fallback, via
Ollama. Why dropped: Qwen and DeepSeek are banned by GSU's AI tool registry, and ARCTIC's gateway
offers much larger models. The local GPU is now used only for Kronos.

## 5. Other design choices that were replaced or rejected

| Dropped | Why or what replaced it |
|---|---|
| Backtesting (even as a stretch goal) | LLMs may have seen news and prices from any past period; the study is live only |
| One fixed optimizer | Five fixed strategies, to see which ones LLM views help or hurt |
| Single LLM score plus a confidence field | Low / likely / high range; width sets uncertainty |
| "Investor mindset" or persona prompts | Called specialized skills files |
| LightGBM in the quant baseline | Not used; baseline is momentum plus Kronos |
| Rerun controls (multiple runs per prompt type) | One run per prompt type |
| Cost-per-token ROI framing | Dropped in favor of skills across strategies |
| Holm on every family | Holm only on pre-registered main comparisons; Benjamini-Hochberg for the rest |
| ETF proxies for the final factor regression | French daily FF5 plus momentum (ETF proxies only for live monitoring) |
| Sharpe without the risk-free rate | Sharpe net of the T-bill rate |
| 1% tie band | Narrower, cost-based band with a sensitivity check |
| 10 references with low relevance in the proposal | Removed 2026-11-11 (`08_literature_and_references.md`) |
