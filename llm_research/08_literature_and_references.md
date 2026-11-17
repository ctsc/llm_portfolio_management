# 08 Literature and references

Last updated: 2026-11-11.

- Part A: the 51 references in the current research proposal, numbered as they appear in
  `current_documents/Project_Proposal_IEEE.pdf`, with what each is for.
- Part B: sources behind the evaluation research (`06_evaluation_research_findings.md`).
- Part C: the 17 references whose abstracts were read on 2026-11-11, and the 10 removed.

Note: the proposal's body has no in-text citation numbers yet; the Related Work paragraph names no
references. Adding them is an open item.

## A. Current proposal references (51)

### LLM views in Black-Litterman and portfolio optimization (closest prior work)
1. Y. Lee, Y. Kim, J. Kim, S. Kim, and Y. Lee, "LLM-Enhanced Black-Litterman Portfolio Optimization", arXiv:2504.14345, 2025. https://arxiv.org/abs/2504.14345 (the study's starting point)
2. L. A. Mantshimuli and J. W. Muteba Mwamba, "Toward a Unified Agentic Framework for Regime-Aware Portfolio Optimization with LLM Signals", Int. J. Data Sci. Anal., vol. 22, Art. no. 92, 2026. https://link.springer.com/article/10.1007/s41060-026-01066-0
3. Z. Li and L. Min, "CVaR-based Portfolio with Investment Views Given by Large Language Models", IAENG Int. J. Appl. Math., vol. 55, no. 8, pp. 2512-2520, 2025. https://www.iaeng.org/IJAM/issues_v55/issue_8/IJAM_55_8_21.pdf
4. L. A. Mantshimuli and J. W. Muteba Mwamba, "Few-Shot Portfolio Optimization: Can Large Language Models Outperform Quantitative Portfolio Optimization?", J. Risk Financial Manag., vol. 19, no. 5, Art. no. 320, 2026. https://www.mdpi.com/1911-8074/19/5/320
20. A. Voronina, O. Romanko, R. Cao, R. H. Kwon, and R. Mendoza-Arriaga, "Generative AI-enhanced Sector-based Investment Portfolio Construction", arXiv:2512.24526, 2025. https://arxiv.org/abs/2512.24526
21. A. Kargarzadeh, N. Khaledian, N. Parvini, and A. Khaledian, "Large Language Model-Driven Small-Capitalization Trading: Integrating Financial News Sentiment, Macroeconomic Indicators, and Technical Signals", arXiv:2608.12283, 2026. https://arxiv.org/abs/2608.12283 (feeds LLM-estimated uncertainty into the covariance used by allocators; finds the allocator matters as much as the sentiment model; relevant to H3b and H4)
22. L. A. Mantshimuli and J. W. Muteba Mwamba, "Enhancing Portfolio Optimization with Multi-LLM Sentiment Aggregation: A Black-Litterman Integration Approach", Investment Manag. Financial Innov., vol. 22, no. 3, pp. 213-226, 2025. https://doi.org/10.21511/imfi.22(3).2025.16
39. Y. Wang, T. G. You, L. Boussioux, and S. Liu, "SOLID: a Framework of Synergizing Optimization and LLMs for Intelligent Decision-Making", arXiv:2511.15202, 2025. https://arxiv.org/abs/2511.15202
43. M. Florencio, "Grounding Investor Views: Neural Predicates in the Black-Litterman Model", arXiv:2607.20533, 2026. https://arxiv.org/abs/2607.20533

### Prompts, skills and personas
5. Y. Kim, Y. Lee, J. Kim, and Y. Lee, "GuruAgents: Emulating Wise Investors with Prompt-Guided LLM Agents", arXiv:2510.01664, 2025. https://arxiv.org/abs/2510.01664
6. X. Li et al., "SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks", arXiv:2602.12670, 2026. https://arxiv.org/abs/2602.12670
19. Y. Abe, S. Matsuo, R. Kondo, and R. Hisano, "Leveraging Large Language Models for Institutional Portfolio Management: Persona-Based Ensembles", arXiv:2411.19515, 2024. https://arxiv.org/abs/2411.19515
33. S. S. Cao, W. Jiang, and H. Xu, "Seeing the Goal, Missing the Truth: Human Accountability for AI Bias", NBER Working Paper 35142, 2026. https://www.nber.org/papers/w35142 (telling an LLM the downstream goal biases its outputs, only for pre-cutoff data)

### LLM return prediction, lookahead and memorization
7. A. Lopez-Lira and Y. Tang, "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models", arXiv:2304.07619, 2023. https://arxiv.org/abs/2304.07619
9. Z. Gao, W. Jiang, and Y. Yan, "Detecting Lookahead Bias in LLM Forecasts", arXiv:2512.23847, 2025. https://arxiv.org/abs/2512.23847
17. Y. Chen, B. T. Kelly, and D. Xiu, "Expected Returns and Large Language Models", SSRN Working Paper 4416687, 2022. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687
18. S. He, L. Lv, A. Manela, and J. Wu, "Chronologically Consistent Large Language Models", arXiv:2502.21206, 2025. https://arxiv.org/abs/2502.21206
23. S. Wu et al., "From Language-Model Stock Rankings to Testable Economic Rules: A Computational Audit", arXiv:2610.01213, 2026. https://arxiv.org/abs/2610.01213
40. N. Deng, J. Wang, Q. Li, J. Zhang, and Y. Yang, "FinRankGRPO: Optimizing LLMs for Listwise Financial Asset Ranking via Group Relative Policy Optimization", arXiv:2609.24175, 2026. https://arxiv.org/abs/2609.24175 (ranks assets, rewarded by Spearman correlation; reports 0.023, a realistic IC scale)

### Live evaluation, benchmarks and evaluation pitfalls
8. C. Li et al., "Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking", Proc. NeurIPS Datasets and Benchmarks Track, 2025. https://arxiv.org/abs/2505.11065
24. Q. Duan et al., "Can Agentic Trading Systems Pay for Their Own Intelligence?", arXiv:2607.10286, 2026. https://arxiv.org/abs/2607.10286
25. J. Chen and G. Wang, "The Price of Thought: Does Test-Time Reasoning Pay in LLM Trading?", arXiv:2609.30705, 2026. https://arxiv.org/abs/2609.30705 (holds portfolio construction fixed; extra reasoning does not reliably raise net returns; supports using no reasoning models)
26. B. Qu and M. Chen, "CLQT: A Closed-Loop, Cost-Aware, Strategy-Consistent Benchmark for Diagnostic Evaluation of LLM Portfolio-Management Agents", arXiv:2606.29771, 2026. https://arxiv.org/abs/2606.29771 (point-in-time data, costs, a four-week live paper test; top capability score was not top Sharpe; after costs agents beat defensive baselines but not reliably the index)
27. T. Zhu et al., "From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets", arXiv:2605.28359, 2026. https://arxiv.org/abs/2605.28359 (masks identifiers, Barra-style attribution into market, style and selection; returns mostly passive exposure once leakage is controlled; supports H5)
28. H. Yu, F. Li, and J. You, "LiveTradeBench: Seeking Real-World Alpha with Large Language Models", arXiv:2511.03628, 2025. https://arxiv.org/abs/2511.03628
29. Y. Chen et al., "StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?", arXiv:2510.02209, 2025. https://arxiv.org/abs/2510.02209
30. W. W. Li, H. Kim, M. Cucuringu, and T. Ma, "Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?", Proc. KDD Datasets and Benchmarks Track, 2026. https://arxiv.org/abs/2505.07078
31. Y. Ye et al., "The Alpha Illusion: Reported Alpha from LLM Trading Agents Should Not Be Treated as Deployment Evidence", arXiv:2605.16895, 2026. https://arxiv.org/abs/2605.16895
32. J. Yao and Z. Zheng, "Beyond Agent Architecture: Execution Assumptions and Reproducibility in LLM-Based Trading Systems", arXiv:2606.08285, 2026. https://arxiv.org/abs/2606.08285
41. Y. Zhao, S. Chen, and N. Su, "PortBench: A Correlation-Aware, Full-Pipeline Benchmark for LLM-Driven Portfolio Management", arXiv:2605.27887, 2026. https://arxiv.org/abs/2605.27887
42. H. Cho and J. H. Kim, "Constructing a Portfolio Optimization Benchmark Framework for Evaluating Large Language Models", arXiv:2603.09301, 2026. https://arxiv.org/abs/2603.09301
44. W. Fu, "The New Quant: A Survey of Large Language Models in Financial Prediction and Trading", arXiv:2510.05533, 2025. https://arxiv.org/abs/2510.05533 (survey covering leakage, portfolio construction and evaluation)

### Quant baseline and portfolio methods
10. Y. Shi et al., "Kronos: A Foundation Model for the Language of Financial Markets", arXiv:2508.02739, 2025. https://arxiv.org/abs/2508.02739
11. F. Black and R. Litterman, "Global Portfolio Optimization", Financial Analysts J., vol. 48, no. 5, pp. 28-43, 1992.
12. O. Ledoit and M. Wolf, "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices", J. Multivariate Anal., vol. 88, no. 2, pp. 365-411, 2004.
13. R. T. Rockafellar and S. Uryasev, "Optimization of Conditional Value-at-Risk", J. Risk, vol. 2, no. 3, pp. 21-41, 2000.
14. D. Goldfarb and G. Iyengar, "Robust Portfolio Selection Problems", Math. Oper. Res., vol. 28, no. 1, pp. 1-38, 2003.
15. R. Michaud and R. Michaud, "Estimation Error and Portfolio Optimization: A Resampling Solution", J. Investment Manag., vol. 6, no. 1, 2008.
16. M. Lopez de Prado, "Building Diversified Portfolios that Outperform Out-of-Sample", J. Portfolio Manag., vol. 42, no. 4, pp. 59-69, 2016. (HRP)
34. H. Markowitz, "Portfolio Selection", J. Finance, vol. 7, no. 1, pp. 77-91, 1952.

### Evaluation and statistics
35. R. C. Grinold, "The Fundamental Law of Active Management", J. Portfolio Manag., vol. 15, no. 3, pp. 30-37, 1989.
36. E. F. Fama and K. R. French, "A Five-Factor Asset Pricing Model", J. Financial Econ., vol. 116, no. 1, pp. 1-22, 2015.
37. W. K. Newey and K. D. West, "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix", Econometrica, vol. 55, no. 3, pp. 703-708, 1987.
38. S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure", Scand. J. Statist., vol. 6, no. 2, pp. 65-70, 1979.
45. O. Ledoit and M. Wolf, "Robust Performance Hypothesis Testing with the Sharpe Ratio", J. Empirical Finance, vol. 15, no. 5, pp. 850-859, 2008.
46. E. F. Fama and J. D. MacBeth, "Risk, Return, and Equilibrium: Empirical Tests", J. Political Econ., vol. 81, no. 3, pp. 607-636, 1973. (added by Claude as the standard reference for the regression; not from the research report)
47. Y. Benjamini and Y. Hochberg, "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing", J. Roy. Statist. Soc. B, vol. 57, no. 1, pp. 289-300, 1995.
48. M. A. Petersen, "Estimating Standard Errors in Finance Panel Data Sets: Comparing Approaches", Rev. Financial Stud., vol. 22, no. 1, pp. 435-480, 2009.
49. V. DeMiguel, L. Garlappi, and R. Uppal, "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?", Rev. Financial Stud., vol. 22, no. 5, pp. 1915-1953, 2009.
50. A. W. Lo, "The Statistics of Sharpe Ratios", Financial Analysts J., vol. 58, no. 4, pp. 36-52, 2002.
51. K. R. French, "Data Library", Tuck School of Business, Dartmouth College. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

## B. Sources behind the evaluation research (read at least at abstract level)

1. Ledoit and Wolf (2008), Robust Performance Hypothesis Testing with the Sharpe Ratio. https://www.econ.uzh.ch/apps/workingpapers/wp/iewwp320.pdf
2. Lo (2002), The Statistics of Sharpe Ratios. https://www.tandfonline.com/doi/abs/10.2469/faj.v58.n4.2453
3. Bailey and Lopez de Prado (2014), The Deflated Sharpe Ratio. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
4. Harvey, Liu and Zhu (2016), ... and the Cross-Section of Expected Returns. https://academic.oup.com/rfs/article/29/1/5/1843824
5. Hansen, Lunde and Nason (2011), The Model Confidence Set. https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA5771
6. Romano and Wolf (2005), Stepwise Multiple Testing as Formalized Data Snooping. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2005.00615.x
7. Hansen (2005), A Test for Superior Predictive Ability. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569
8. Benjamini and Hochberg (1995), Controlling the False Discovery Rate. https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x
9. Petersen (2009), Estimating Standard Errors in Finance Panel Data Sets. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1320561
10. DeMiguel, Garlappi and Uppal (2009), Optimal Versus Naive Diversification. https://econpapers.repec.org/RePEc:oup:rfinst:v:22:y:2009:i:5:p:1915-1953
11. Fama and French (2015), A Five-Factor Asset Pricing Model. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2287202
12. Kenneth R. French Data Library (daily 5 factors, daily momentum, RF). https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
13. Ding and Martin (2017), The Fundamental Law of Active Management: Redux. https://www.sciencedirect.com/science/article/pii/S0927539817300543
14. Newey and West (1994), Automatic Lag Selection in Covariance Matrix Estimation. https://academic.oup.com/restud/article-abstract/61/4/631/1590368
15. Politis and White (2004), Automatic Block-Length Selection for the Dependent Bootstrap; Patton, Politis and White (2009) correction. https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf and https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf
16. Memmel (2003), Performance Hypothesis Testing with the Sharpe Ratio. https://books.google.com/books/about/Performance_Hypothesis_Testing_with_the.html?id=BDLpzgEACAAJ
17. Opdyke (2007), Comparing Sharpe Ratios: So Where Are the p-values? http://www.datamineit.com/Sharpe%20Ratio%20Comparisons%20-%20J.D.%20Opdyke%20-%20preprint%20-%20Journal%20of%20Asset%20Management,%20Vol8(5),%202007.pdf
18. Diebold and Mariano (1995), Comparing Predictive Accuracy. https://ideas.repec.org/a/bes/jnlbes/v13y1995i3p253-63.html
19. Harvey, Leybourne and Newbold (1997), Testing the Equality of Prediction Mean Squared Errors. http://eurequa.univ-paris1.fr/membres/Ahamada/cours/article3.pdf
20. Lopez-Lira and Tang, Can ChatGPT Forecast Stock Price Movements? https://arxiv.org/abs/2304.07619
21. Lopez-Lira, Tang and Zhu, The Memorization Problem: Can We Trust LLMs' Economic Forecasts? https://arxiv.org/abs/2504.14765
22. Evaluating LLMs in Finance Requires Explicit Bias Consideration (2026). https://arxiv.org/html/2602.14233v1
23. Shi et al. (2025), Kronos. https://arxiv.org/abs/2508.02739
24. He et al., Chronologically Consistent Large Language Models (ChronoGPT). https://arxiv.org/pdf/2502.21206

Candidates to add to the proposal if Related Work or Evaluation needs them: Romano and Wolf
(2005), Hansen (2005), Hansen, Lunde and Nason (2011), Bailey and Lopez de Prado (2014), Harvey,
Liu and Zhu (2016), Lopez-Lira, Tang and Zhu, and the bias-consideration paper.

## C. The 17 abstracts read on 2026-11-11 and the 10 removed

The 17 were first picked as low-value candidates from titles alone. After reading each abstract
(as summarized by the fetch tool, not word for word), 10 were removed and 7 kept. Several title-based
judgments were wrong, which is why the abstracts were read.

### Removed (10)

| Paper | Abstract in short | Why removed |
|---|---|---|
| TradingAgents (Xiao et al., arXiv:2412.20138) | Multi-agent firm simulation: analyst roles, bull/bear debate, risk team, traders | No link to optimizer views |
| FinMem (Yu et al., arXiv:2311.13743) | Agent with profiling, layered memory and decision modules | Agent memory design |
| AlphaAgents (Zhao et al., arXiv:2508.11152) | Role-based agents for stock selection at different risk tolerances | GuruAgents already covers specialized prompts |
| AlphaCrafter (Yuan et al., arXiv:2605.05580) | Structured "agent harness" for multi-agent trading | Agent workflow engineering |
| Regret-Driven Portfolios (Abro and Jaleel, arXiv:2601.17021) | Online learning with sentiment filters and LLM hedging | Different method |
| Kirtac and Germano, REALM 2025 | LLM sentiment added to PPO reinforcement learning on a 3-asset portfolio | Reinforcement learning, tiny portfolio |
| TraderBench (Yuan et al., arXiv:2603.00285) | Adversarial crypto and options agent tasks | Off-domain |
| AI-Trader (Fan et al., arXiv:2512.10971) | Live agent benchmark across markets | Overlaps with LiveTradeBench and DeepFund |
| The Inefficient Pricing of News (Didisheim et al., NBER w35093) | Prices underreact to news text; "pure news" predicts returns | Not about LLM views or portfolios |
| When Agents Trade (Qian et al., arXiv:2510.11695) | Live multi-market agent arena; agent design shaped behavior more than the model | Closest call; about agent frameworks, not prompts |

### Kept after reading (7)

From Knowing to Doing; Cao, Jiang and Xu; the small-cap trading paper; The Price of Thought; CLQT;
FinRankGRPO; The New Quant survey. Reasons are in the annotations in part A.

Next candidates if more need to be cut: SOLID (39), Duan et al. (24), Yao and Zheng (32).
