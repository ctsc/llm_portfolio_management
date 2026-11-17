# 12 AI hallucination research (earlier project, dropped 2026-10-20)

Last updated: 2026-11-11. This is NOT the current project. It records Carter's earlier MS research
direction, which was replaced by the LLM portfolio study. Sources: Carter's saved notes and an
earlier chat (2026-09-22, "AI hallucination statistics in academic papers"). The papers named
below were discussed in that chat and have not been re-checked for this file.

## Timeline

- September 2026: started MS research on AI hallucination in academic and legal papers, advisor Dr.
  Parag Tamhankar (the topic was originally Parag's suggestion: detecting AI hallucination and the
  confidence level of LLM output).
- Legal papers dropped ("too much"); academic papers only.
- Several redesigns (below). By 2026-10-18 the research had no set direction.
- 2026-10-20: dropped for the LLM stock study, because hallucination already seemed extensively
  covered by existing research and tools. Carter planned to tell Parag about the change.

## How the design evolved

1. A general hallucination and confidence detector. Confidence scored differently by question type:
   factual claims graded against verifiable truth; opinion or semantic claims scored by consistency.
   Leaned toward engineering and training his own model rather than a harness around frontier
   models. Rejected a code or package-hallucination framing (coding models are already strong).
2. Inspired by TypeSafe AI's "Jev" (a non-autoregressive model that takes structured state and
   returns typed decisions with calibrated confidence in one pass): a JSON-in, typed-decision-out
   fusion layer on top of a multi-signal verification pipeline, released as a Hugging Face
   checkpoint. Harm framing: academic retractions, legal citation hallucinations, and a 2026 military
   case where an analyst's chatbot hallucinated that a ship carried nuclear program components.
3. A citation-hallucination verifier modeled on "Source or It Didn't Happen" for academic citations,
   extended to legal citations, with calibrated confidence. Build in stages: recreate CiteTracer plus
   a claim-support and confidence model on academic papers first, then legal. Main contribution aimed
   at the false-positive problem (real citations flagged as fabricated) seen in a legal citation
   benchmark. Uses real papers and filings flagged for hallucinated citations.
4. Two models: citation existence (largely solved, secondary) and claim support (main focus): given a
   real cited source, does it actually contain the attributed claim? A claim can be true and the
   source real, yet the claim not be in that source. Combine regex or exact matching for verbatim
   quotes with a fine-tuned model.
5. Carter felt this had become "just a fact checker". Moved toward an OpenScholar-style system (Ai2
   and UW, retrieval-grounded scientific citations) where citations are chosen by a typed model he
   trains ("Typed Fine-Tuning", TFT; he disliked the name "picker"). Contribution framed as
   writer-agnostic: the typed citation model improves citations across several existing LLMs (for
   example Llama, Qwen, GPT-4o, OpenScholar) rather than beating OpenScholar head to head. Search
   through OpenScholar's released library (computer science slice first; full library a stretch goal
   if GSU HPC allows); Semantic Scholar only for checking whether a cited source exists. Start from an
   existing open LLM writer and fine-tune only if needed.

## Prior art found (from the 2026-09-22 chat)

| Line of work | Examples | What it does |
|---|---|---|
| Detect, search and revise | RARR (Gao et al., Google and CMU, 2022), SAFE (DeepMind), RAC | Generate questions about a claim, search the web, edit the text to match evidence |
| Claim verification inside a cited source | SciFact (Wadden et al., 2020), WICE, VerifAI, DeepSciVerify (2026) | Check a claim against the specific cited document (support, refute, not enough info); DeepSciVerify escalates from abstract to full text |
| Citation auditing | GhostCite, CiteAudit, a Reference Hallucination Score | Check whether citations exist; mostly built around frontier models |
| Commercial legal tools | Lexis+, Westlaw | Retrieval-based; even RAG systems still fabricate some citations |

Statistics sources mentioned in that chat: Walters and Wilder (Scientific Reports, 2023), Topaz et
al. (The Lancet, 2026), GhostCite (2026), Turnitin detection reports, Stanford's legal
hallucination studies.

## Gaps identified at the time (for reference)

- Most systems wrap existing frontier models instead of training a purpose-built decision model.
- Work is citation-specific or domain-specific, not general.
- Verdicts are generated as text, token by token; no single-pass typed decision.
- Confidence is not split by factual vs opinion claims.
- Retrieval-augmented approaches plateau.
- Large drop from synthetic benchmarks to real cases (for example CiteAudit about 97% to 90%).
- Mostly single-signal checks rather than fusion.
- No system spans academic, legal and defense contexts.
- Few released model weights.
- Binary labels rather than calibrated, structured scores.

The more defensible angle at the time: use a trained typed classifier to decide which claims are
worth verifying (instead of checking all of them), extend single-source verification across full
documents and domains, and output calibrated confidence instead of three-way labels.

## Why it was dropped

Hallucination detection and citation verification already had extensive research and tools, so
novelty was hard to defend. On 2026-10-18 Carter considered an agentic loop that manages a portfolio
with research and skills (inspired by the ai-berkshire GitHub project) and measures cost-adjusted
returns per model. On 2026-10-19 that narrowed to the current study: how effective specialized skills
are in LLM-Enhanced Black-Litterman portfolio optimization (Lee et al., arXiv:2504.14345), compared
with a quant baseline (`01_project_overview.md`). The cost-per-token framing was dropped.
