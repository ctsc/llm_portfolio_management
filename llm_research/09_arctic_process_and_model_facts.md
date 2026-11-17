# 09 ARCTIC process, gateway and model facts

Last updated: 2026-11-17. Everything here was read on 2026-11-11 from the pages listed at the end.
Items marked UNKNOWN are not in any public page and go to Andalib (`manualwork.md` section A).
Estimates are labeled as estimates.

## A. How access works (ARCTIC wiki)

The wiki has two separate routes, and no page says which one an LLM-only user needs.

| Route | What the wiki says | Source |
|---|---|---|
| Gateway key | Open a Hydra ticket, name the models, and ARCTIC staff send the API key through a 1Password one-time link. No mention of allocations, service units or proposals on either LLM page. | LiteLLM/api, LiteLLM/OpenCode |
| Allocation request | Log into Elpis, click the project, "Request Resource Allocation", fill the abstract, select users, attach the supporting documents. | home/elpis/additional-allocation |

Andalib's email of 2026-10-28 pointed Carter at the allocation route and asked that the request
name the models, the use case, the compute and the duration. Whether gateway-only use still needs an
allocation is UNKNOWN and is question 1 of the email to Andalib.

## B. Allocation rules that matter for this request

| Rule | Detail |
|---|---|
| GPU service unit | One NVIDIA V100 32 GB for one hour. Other GPUs are scaled by power. This is the "V100-equivalent" unit used in our request. |
| Startup allocation (SRCA) | One per GSU researcher per period. Capped at 10,000 CPU SU or 1,000 GPU SU. 150 active jobs, 200 GB, preemptible. Not renewable. The Elpis startup page says approval grants 10,000 CPU hours on ACIDS. |
| Research allocation (ARCA) | PI (Parag) submits. Reviewed quarterly by the Resource Allocation Committee. 12-month period, unused SU are forfeited. Not preemptible. |
| ARCA windows | Jan 15 to Feb 1 (notified Mar 15, begins Apr 1); Apr 15 to May 1 (Jun 15, Jul 1); Jul 15 to Aug 1 (Sep 15, Oct 1); Oct 15 to Nov 1 (Dec 15, Jan 1). Today is Nov 11, so the October window has closed. |
| Sizing basis | "Calculations should be based on small-scale test runs on the resources of interest and scaling tests, as well as the number and type of runs." Our 40 GPU-seconds per request is an estimate, not a test run. |
| Paid allocation | Immediate, billed monthly; V100 $0.40, A2000 $0.20, RTX $0.10, TITAN $0.14 per GPU-hour. Whether it applies to the gateway is UNKNOWN. |

Consequences, stated as facts and not as decisions:
- The 300-stock request (2,730 GPU-hours since the 2026-11-12 revision) is above the 1,000
  GPU-hour startup cap. The 100-stock request (910) is below it.
- The next regular ARCA window (Jan 15 to Feb 1) leads to an April 1 start, which is after the
  end of November date Carter needs access by.
- Which route applies, and whether the gateway is metered at all, decides whether any of this
  matters. That is why question 1 and question 2 of the email come first.

## C. Required documents and formatting (policy/resource-allocations)

| Document | Rule |
|---|---|
| Main Document | Required, 3 pages maximum. Over the limit means the submission is returned unreviewed. |
| Code Performance and Scaling | Required, 2 pages maximum. Our current request does not include one. The sample (Gozem) is built from measured run times and a per-calculation SU table. |
| References | Optional, separate. |
| Progress report, publications | Renewals and supplements only. |
| Format | PDF only, 1-inch margins, Times New Roman 11 pt or larger (Arial, Courier New, Palatino 10 pt or larger), at most 15 characters per inch and 6 lines per vertical inch, page numbers in each file. |

Our `ARCTIC_Resource_Request.pdf` meets the Main Document rules (3 pages, Times 11 pt, 1-inch
margins, page numbers). A Code Performance and Scaling document would have to be written from
measured gateway requests (tokens in, tokens out, seconds, finish reason), which need an API key
first.

## D. The LiteLLM gateway (LiteLLM/api, read 2026-11-11)

| Item | Fact |
|---|---|
| Status | Alpha. "Models may be slow or unavailable without notice, and models and settings may change. Don't rely on them for deadlines or critical work." |
| Base URL | `https://litellm.rs.gsu.edu/v1`, OpenAI-compatible. Health check: `/health/liveliness` returns "I'm alive!". |
| Network | GSU campus, GSU VPN or an ARCTIC login node, for every request (rechecked 2026-11-17): "Get access" says to connect from the GSU network, and the Troubleshooting table says a timed-out or refused connection means you are not on the GSU network and to connect to the VPN or use a login node. Every API call goes to the same host, `litellm.rs.gsu.edu`. |
| Auth | `Authorization: Bearer <key>` or an `x-litellm-api-key` header. Key may be limited to some models. |
| Model IDs | `google/gemma-4-31B-it`, `MiniMaxAI/MiniMax-M2.7`, `BAAI/bge-m3` (embeddings, not needed). |
| Endpoints | `/v1/chat/completions`, `/v1/responses`, `/v1/messages`. `/v1/completions` returns meaningless text. No images. |
| JSON output | `response_format` with `json_object` or `json_schema` (strict) is documented and was tested. Our config still says `json_mode: true  # UNVERIFIED`. |
| Seed | Not on the wiki. Andalib (2026-11-12): passed through to the model server, but outputs are not guaranteed identical. Our config said `send_seed: true  # UNVERIFIED` before that. The client already drops `seed` if the gateway rejects it. UNKNOWN whether it is honored. |
| Temperature | Accepted (their example uses 0.2). |
| Errors | 429 "No deployments available ... cooldown" means wait a few seconds and retry. |
| Logging | "ARCTIC records details of each request, such as your user ID, the time, the model, and how much you used. It may also keep prompts and responses." Inputs and outputs may be public records under the Georgia Open Records Act. |
| Data | Approved for unregulated university data only. Our inputs are public prices and news. |
| Policy | Covered by GSU's AI policy. Researchers follow URSA's guidance on AI in research and disclose AI use. |
| Tested with | LiteLLM 1.104.0 on 2026-11-06 from a Mac on the VPN and from an ARCTIC login node. |

Rate limits, token caps per key, concurrency limits, VPN session limits and the number of GPUs behind
each model are not documented. They are questions 3 and 7 of the email to Andalib.

## E. The two models

| | Gemma 4 31B (`google/gemma-4-31B-it`) | MiniMax M2.7 (`MiniMaxAI/MiniMax-M2.7`) |
|---|---|---|
| Size | 30.7B parameters, dense | 230B total, 10B active (mixture of experts) |
| Context | 256K tokens | 196K tokens per sequence |
| License | Apache 2.0 | "other" (license file in the repository) |
| Precision | BF16 (about 60 GB of weights, my inference) | Native FP8, about 220 GB of weights |
| Serving reference | Not stated | 4 high-memory GPUs (A100 or H100 class) with tensor parallel 4; the vLLM recipe default is 8 H200 |
| Thinking | Off unless `<\|think\|>` is added to the system prompt. With thinking off, the 31B model still emits an empty `<\|channel>thought` block before the answer. | Always thinks. Reasoning returns in `reasoning_content`. The answer can start with blank lines. |
| Effect on `max_tokens` | None while thinking is off | Thinking counts toward `max_tokens`. If the cap is too low, content is empty (None) with `finish_reason: length`. ARCTIC measured about 100 tokens for a one-word answer and suggests about 2,000. |
| Model card sampling | temperature 1.0, top_p 0.95, top_k 64 | temperature 1.0, top_p 0.95, top_k 40 |
| Can thinking be disabled or capped | Yes (do not enable it) | Not stated anywhere read, including the vLLM recipe, MiniMax's SGLang guide and the model card. UNKNOWN on ARCTIC's deployment. |

What these facts change in the study (each is in PLAN4):

1. **MiniMax output cap.** `max_output_tokens: 300` returns empty answers with `finish_reason: length`.
   MiniMax needs its own cap (about 2,000 to start), and the invalid-rate rule (15%) would otherwise
   send every MiniMax week to the Q0 fallback. The ARCTIC request's "300 output tokens" is wrong for
   MiniMax.
2. **Gemma channel block (corrected 2026-11-11).** Gemma's `<|channel>thought ... <channel|>` marker
   does not break parsing: `parse_response` takes the first JSON object anywhere in the text, so the
   marker is skipped. An earlier note here said the answer would be marked invalid; that was wrong.
3. **Temperature.** Both model cards recommend 1.0. The study uses 0.0 (DECISION D4) for repeatability.
   The decision is open for both models and is in `manualwork.md` section C. If kept at 0, the paper
   says why.
4. **GPU-second basis.** 40 V100-equivalent GPU-seconds per request has no source. A rough
   FLOP-based check (2 x parameters FLOPs per token, 40% utilization, 4 requests batched, 8,000
   tokens in, 300 out) gives about 15 GPU-seconds for Gemma on a V100, about 6 on an A100 and about
   2.6 on an H100. MiniMax with about 2,000 reasoning tokens could be much higher and was not
   estimated reliably. These are estimates only. The real figure needs Andalib's answer on how
   gateway use is metered plus measured requests.
5. **Input tokens.** The bare prompt is about 270 tokens (`base_instructions.md`, 1,068 bytes), news
   adds 0 to 1,500 and a skills file up to 6,000 (C4), so one request is about 300 to 7,800 input
   tokens, not 2,000 to 8,000.

## G. Andalib's answers (2026-11-12)

From his reply to the seven questions (full text in the gitignored
`llm_research/correspondence/andalib_reply_2026-11-12.md`).

| Topic | Answer |
|---|---|
| Access route | No formal process for gateway-only access yet; his team is deciding how to handle it. He knows access is needed by the end of November. Meanwhile, open a Hydra ticket at https://hydra.gsu.edu, assigned to `arctic_developers`, with the request PDF attached. |
| Metering and limits | Being decided by his team (how use is counted, what limits each key has). |
| Offer | Anything dropped only because of timing (small models, Slurm resources) may be accommodated. Carter declined both (2026-11-12). |
| Tokens | He asked which input range was right (the email's 300 to 8,000 was). MiniMax's reasoning tokens count as output, so the estimate should use about 2,000 output tokens for it. The PDF was revised on 2026-11-12. |
| MiniMax thinking | As far as they know, no setting turns it off. He is checking how their deployment returns the reasoning. |
| Seed | The gateway passes `seed` to the model server. Outputs are not guaranteed identical even at temperature 0, because requests are batched with other users'. Plan for small differences and measure them by rating a fixed set of stocks twice in the pilot. |
| Stability | Alpha service, no status page or changelog. Log the `model` field from each response and run a small fixed set of canary prompts at the start of each weekend; mark a week whose outputs shift. |
| VPN | Run by GSU IT, not ARCTIC; ask the IT help desk about session limits. Save each rating as it completes and resume after a dropped connection. |
| Data | Public data only is the right choice, since the gateway may keep prompts and responses. |

Follow-ups from Andalib (2026-11-12 to 2026-11-16):

| Topic | Answer |
|---|---|
| Access route | Parag requests access through Elpis (https://elpis.rs.gsu.edu) and attaches the revised PDF as a supporting document. Once the project is set up, ARCTIC can start providing access. No Hydra ticket is needed (Carter's Hydra login failed; Andalib suggested resetting the GSU password if it is ever needed). |
| Usage numbers | "They look good." How usage is counted is still being decided. |
| MiniMax reasoning | Returned in a separate `reasoning_content` field; `content` holds only the final answer, sometimes with a few leading newlines. Each response's `usage` reports `reasoning_tokens`. |
| Model versions | The gateway returns no model version or `system_fingerprint`; `/v1/models` lists names only. Each response has the `model` field, an `x-litellm-model-id` header (the gateway's deployment ID; may change when a model is reconfigured, but will not catch every change) and an `x-litellm-version` header (gateway software version). `/v1/model/info` has an `updated_at` per model that changes when gateway settings are edited, not when the model server behind it changes, and also for changes that do not affect outputs, such as pricing. He still recommends the canaries. |
| VPN | GSU IT help desk: help@gsu.edu or 404-413-HELP (4357). |

Andalib, 2026-11-17 (summary in the gitignored `llm_research/correspondence/andalib_reply_2026-11-17.md`):

| Topic | Answer |
|---|---|
| Hydra | Still needed: after the Elpis allocation, open a Hydra ticket requesting LiteLLM access and mention the project team. This replaces the 2026-11-12 "no Hydra ticket" note, which applied to the allocation request only. He suggested resetting the GSU password to fix the Hydra login. |
| LiteLLM project | The PI (Parag) creates the project in LiteLLM, then adds Carter to it. Without it there is no team to create keys under, which is why the "Create New Key" button is missing. |
| Spending limit | Probably $10 to start, raised if it is hit. This is the gateway's own dollar budget, separate from the 1,000 GPU service units in Elpis. What one request costs in gateway dollars is unknown until the first responses (`x-litellm-response-cost`). |
| VPN | Required to call the models, not only for the UI. |
| Other limits | Rate, per-call token and concurrency limits were not stated. |

Allocation granted (2026-11-16): the default startup allocation for GSU users, 1,000 GPU service
units and 10,000 CPU service units, metered as GPU and CPU usage. Carter has the allocation's Slurm
account name and value (kept in his own records, not here). Startup allocation rules from the policy
page: capped at 1,000 GPU or 10,000 CPU service units, 150 active jobs, 200 GB project space, jobs
preemptible, not renewable (a research allocation request is the path to more). At the request's
estimates, 100 stocks needs about 607 GPU-hours over 24 weeks, which fits; 300 stocks (about 1,820)
does not, so the study uses 100 stocks (B4).

Approval (2026-11-16): Carter is approved in Elpis. The request PDF is attached to the allocation
(supporting documents go on allocations, not projects), and Carter was added to the project as a
Manager, so he can manage it in Elpis without Parag. Parag would need to complete ARCTIC's course
only if he plans to use ARCTIC resources himself. The approved amount, metering and key limits were
not stated.

Getting a key (the wiki's "Get access" section as of 2026-11-17; it replaced the earlier "staff
send the key through 1Password" process):
1. Be on the GSU network or VPN (`curl -sS https://litellm.rs.gsu.edu/health/liveliness` prints
   "I'm alive!").
2. Be on an approved ARCTIC project in Elpis (done).
3. Sign in once at https://litellm.rs.gsu.edu/ui with the GSU campus ID and password so the account
   exists.
4. Parag creates the project in the LiteLLM UI and adds Carter (Andalib, 2026-11-17). Open a Hydra
   ticket requesting LiteLLM access, naming the project and the models; ARCTIC staff confirm. Once
   per project. (Carter's Hydra login fails; reset the GSU password first.)
5. Create the key: sign in at https://litellm.rs.gsu.edu/ui, Virtual Keys, "+ Create New Key",
   Team = the project, a descriptive name, Models left empty (all project models) or only the ones
   needed, "Create Key", and copy it at once (shown only once) into a password manager.
6. One key per tool or script, so one can be replaced without breaking the others; regenerate a key
   on the Virtual Keys page if it may have leaked. Keep keys out of code and repositories. The wiki
   loads the key as `LITELLM_API_KEY`; this repository reads `ARCTIC_LITELLM_API_KEY` from `~/.env`.

What it changed: temperature 0 confirmed (D4), the pilot repeat check (G20), weekly model checks
and canaries (G21), the revised request numbers (G17), the Elpis submission and VPN steps in
`manualwork.md` section A, and the logged identity fields in G21 (deployment ID and gateway version
headers, `/v1/model/info`, `reasoning_tokens`).

## F. Sources read on 2026-11-11

- ARCTIC: [Use ARCTIC's AI models from your code](https://arcwiki.rs.gsu.edu/en/LiteLLM/api)
- ARCTIC: [Using OpenCode with ARCTIC's AI Models](https://arcwiki.rs.gsu.edu/en/LiteLLM/OpenCode)
- ARCTIC: [Resource Allocations](https://arcwiki.rs.gsu.edu/policy/resource-allocations)
- ARCTIC: [Requesting Additional Resources with Elpis](https://arcwiki.rs.gsu.edu/en/home/elpis/additional-allocation)
- ARCTIC: [Using Elpis to Request a Startup Allocation](https://arcwiki.rs.gsu.edu/en/home/elpis/startup-allocation-gsu)
- ARCTIC: [Node configuration](https://arcwiki.rs.gsu.edu/en/hardware/node-config) (V100 32 GB, A100 40 GB, A40 48 GB, A30 nodes exist; which serve the gateway is not stated)
- ARCTIC: sample [Main Document](https://arcwiki.rs.gsu.edu/elpis/proposal.pdf) and [Code Performance](https://arcwiki.rs.gsu.edu/elpis/code_performance.pdf)
- [Gemma 4 31B model card](https://huggingface.co/google/gemma-4-31B-it)
- [MiniMax M2.7 model card](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)
- [vLLM recipe for MiniMax-M2.7](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M2.7)
- [MiniMax M2.7 deployment guide](https://platform.minimax.cn/docs/guides/local-deploy-m2-7.md)
- [Alpaca market data overview](https://docs.alpaca.markets/docs/about-market-data-api) (for the IEX feed finding in `03`)
