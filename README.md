# Free-model aggregator

One server (`main.py`, Python 3.13, standard library only) that serves the
OpenAI Chat Completions API and the Anthropic Messages API — each in JSON and
in server-sent events — over the ten free-tier providers a trial's brief
names, and answers every request with the best free model that can serve it.

## How it works

- **Brief.** The three JSON documents in `$MAKEFUN_PROMPT_FILE` (provider
  directory, quality table, workload manifest) are the only inputs; nothing
  provider-specific is hardcoded, so the same code runs against the real
  services given their keys and base URLs.
- **Catalog.** At startup the server probes every provider's own model list
  in that provider's wire (OpenAI, Anthropic or Gemini), merges each listing
  with the quality table's free flag and context length, and builds one
  catalog document. That document is what `--list` prints and what the
  register body carries — the same object, byte for byte.
- **Routing.** The client sends the workload in manifest order, so each
  arriving request takes the next manifest row's requirements. Candidates
  are the brief's free models meeting those requirements (tools, vision,
  JSON mode, minimum context) that the provider's own listing confirms as
  free, ranked by `rating_milli` (ties: directory order, then model id).
  A JSON-mode OpenAI request is never routed to the Anthropic wire, which
  cannot carry `response_format`.
- **Translation.** Same-wire requests pass through with only the model
  swapped; cross-wire requests are rebuilt field by field — system prompt,
  messages (text and images, byte-identical), tools, tool choice,
  `max_tokens`, stop sequences — so the provider parses and echoes exactly
  what the client sent. Tool calls and tool results are converted between
  the OpenAI, Anthropic and Gemini shapes.
- **Failover.** Candidates are tried best-first; a refusal (402, 429, 5xx or
  a transport failure) moves to the next candidate. Upstream calls are made
  non-streamed and the client's answer is rendered from the whole answer
  text, in JSON or as an SSE stream in the client's wire, with the
  provider's usage counts and its stop reason mapped into the client's
  vocabulary (`end_turn` ↔ `stop`, `max_tokens` ↔ `length`,
  `tool_use` ↔ `tool_calls`, `refusal` ↔ `content_filter`).
- **Lifecycle.** Probe, listen on an OS-assigned port (never 8080),
  `POST $MAKEFUN_ENV_URL/register`, serve, poll `GET $MAKEFUN_ENV_URL/work`
  every 100 ms, exit 0 when it says the workload is done.
- **Determinism.** No wall clock and no randomness reach any answer:
  response ids come from the request counter, timestamps are constants,
  rankings are total orders, and stream chunking is fixed-size. A replayed
  trial leaves a byte-identical state export (verified locally against the
  kit's practice instances).

## Local check

With the kit's env seeded from a practice instance and this server run
against it, the kit's scorer reports 99/100 on each of practice trials
0–2 (served, wire, failover and catalog at 100; quality within a point of
the budget-free bound, the difference being the top provider's spent
free-tier budget, which the bound ignores by design).
