# 13 LiteLLM gateway reference (what matters to us as gateway users)

Last updated: 2026-11-17. ARCTIC's gateway runs LiteLLM, an open-source proxy. Main documentation:
https://docs.litellm.ai/docs/ (API reference: https://docs.litellm.ai/api-reference/). Most of it is
written for whoever runs the gateway; the notes below are the user-side parts. ARCTIC's own wiki
(https://arcwiki.rs.gsu.edu/en/LiteLLM/api) wins wherever the two differ: ARCTIC says some options in
LiteLLM's docs are not supported on its service. Facts here come from LiteLLM's docs, not from ARCTIC;
anything ARCTIC has configured differently is unknown until we see it in real responses.

## Virtual keys, teams and limits

Source: https://docs.litellm.ai/docs/proxy/virtual_keys and https://docs.litellm.ai/docs/proxy/users

- A virtual key is a key issued by the proxy. ARCTIC users create their own in the UI
  (https://litellm.rs.gsu.edu/ui, Virtual Keys, "+ Create New Key", Team = the project).
- A key can carry `models`, `max_budget` with `budget_duration`, `rpm_limit` (requests per
  minute), `tpm_limit` (tokens per minute), `max_parallel_requests`, `expires` and a `team_id`.
- A key with a `team_id` is governed by the team's budget and limits (team `rpm_limit`,
  `tpm_limit`, `max_parallel_requests`, per-model `model_rpm_limit` and `model_tpm_limit`), not by
  the key owner's personal budget.
- `/key/info` shows a key's spend and limits, but ARCTIC says management endpoints such as
  `/key/info` are not available to user keys. The UI and the response headers below are how we see
  our limits.

## Errors to expect

- Rate limit exceeded (key or team `rpm_limit`, `tpm_limit`): HTTP 429. Retry after a wait.
- Budget exceeded: error text such as `ExceededBudget` or `ExceededTokenBudget`, or type
  `budget_exceeded`; the status code varies (400, 401 or 429 in LiteLLM's examples). Retrying
  does not help; it needs ARCTIC.
- A deployment in cooldown after repeated failures: ARCTIC's wiki shows 429 "No deployments
  available for selected model ... cooldown"; wait a few seconds and retry. LiteLLM's cooldown
  settings are `allowed_fails` and `cooldown_time` (example 30 seconds); ARCTIC's values are unknown.
- Unsupported parameter: by default LiteLLM raises an error rather than silently dropping it,
  unless the gateway sets `drop_params`.

## Response headers worth logging

Source: https://docs.litellm.ai/docs/proxy/response_headers

| Header | Meaning | Use for us |
|---|---|---|
| `x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-requests`, `x-ratelimit-reset-tokens` | The key's rate limits and what is left, when the key has limits | Learn our key's limits from real responses (ARCTIC has not stated them) |
| `x-litellm-model-id` | Deployment that served (or failed) the request | Model-change check (G21) |
| `x-litellm-model-group` | The model name the request was routed to | Confirms the requested model answered |
| `x-litellm-attempted-fallbacks`, `x-litellm-fallback-reason` | Number of fallback attempts and why | A fallback means another deployment or model may have answered; must be flagged |
| `x-litellm-attempted-retries`, `x-litellm-max-retries` | Retries inside the gateway | Explains slow requests |
| `x-litellm-response-duration-ms`, `x-litellm-overhead-duration-ms` | Time inside the gateway, and the gateway's own overhead | Server-side timing for the ARCTIC usage estimate, separate from VPN delay |
| `x-litellm-response-cost`, `x-litellm-key-spend` | Cost of the call and the key's total spend, if ARCTIC sets prices | Usage tracking (N7) if ARCTIC meters by cost |
| `x-litellm-version` | Gateway software version | Logged as `gateway_changed` information (G21) |
| `x-litellm-call-id` | ID of the request | Quote it if we report a problem to ARCTIC |

## Reasoning (MiniMax M2.7)

Source: https://docs.litellm.ai/docs/reasoning_content

- LiteLLM standardizes the model's thinking into `message.reasoning_content` (Andalib confirmed this
  for MiniMax). The count is in `usage` (Andalib: `reasoning_tokens`).
- `reasoning_effort` (values "none" to "max") and `max_completion_tokens` (visible output plus
  reasoning) exist as parameters, but whether ARCTIC's MiniMax honors `reasoning_effort` is unknown;
  Andalib said the model has no setting that turns thinking off. A one-off test in the timing run
  could check whether `reasoning_effort: "low"` shortens MiniMax's reasoning; it is not part of the
  study unless Carter decides so before the freeze.

## Request parameters

Source: https://docs.litellm.ai/docs/completion/input

- `temperature`, `top_p`, `seed` (best effort; LiteLLM says determinism is not guaranteed),
  `max_tokens`, `max_completion_tokens`, `response_format` (`{"type": "json_object"}`; the prompt
  must also ask for JSON or the model may run to the token limit), `stop`, `n`, `timeout`.

## Fallbacks (why they matter)

Source: https://docs.litellm.ai/docs/proxy/reliability

- A gateway can be configured to retry on another deployment or fall back to another model group
  when a model errors or is rate-limited. If that happens, a request for one model could be answered
  by a different deployment or model. The headers `x-litellm-model-group`,
  `x-litellm-model-id` and `x-litellm-attempted-fallbacks` show it. Whether ARCTIC uses fallbacks is
  unknown, so the pipeline must check every response (PLAN4 G15 and G21).
