You are rating one stock for a one-week holding period that starts at the next Monday open and ends at the following Monday open.

You will be given the ticker, the company name, the GICS sector, and the news published during the past week. You have no other data, no tools, and no web access. Use only what is in this message.

Rate the stock's expected one-week return relative to the other large US stocks in the same market.

Respond with JSON only, exactly in this form and nothing else:

{"low": <integer -5..5>, "likely": <integer -5..5>, "high": <integer -5..5>, "rationale": "<short reason>"}

Rules for the fields:
- low: the lowest plausible rating for the next week.
- likely: the most likely rating for the next week.
- high: the highest plausible rating for the next week.
- Each rating must be an integer from -5 (strongly underperform) to 5 (strongly outperform). 0 means no view.
- Order the ratings so low <= likely <= high.
- rationale: a short reason, at most 150 tokens.

Do not include code fences, markdown, or any text outside the JSON object.
