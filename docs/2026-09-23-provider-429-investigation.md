# Provider 429s on the HF router — investigation, 2026-09-23

Two live runs of `uv run my-agent` at `LIVE_CHECK_REPEATS=5` each lost an attempt to
`429 ... 'code': 'queue_exceeded'`. This is what the logs say, what was ruled out, and what the
options are.

**Status, 2026-09-23:** the diagnosis held and the pinning experiment ran — see §8a. Three changes
landed: the error-header capture (§3), the `_model_route` fix for a bug pinning exposed (§8a), and
nothing else. `DEFAULT_MODEL` is still the bare id, so the harness still defaults to `:fastest`.

Logs: `logs/20260923T161354Z-81b4f402.jsonl`, `logs/20260923T162954Z-eed4d965.jsonl`.

## Verdict

**The retry configuration is correct and was never the problem. The router explicitly refuses the
retry, and the cause is the routing policy we never chose.**

Omitting a provider suffix from the model id selects HF's `:fastest` policy. Eleven providers serve
`openai/gpt-oss-120b`; `:fastest` concentrates us on the one or two with the lowest first-token
latency, which is where every other caller on the same default also lands, which is where the
queues fill. `queue_exceeded` at roughly one request per second is not our load — it is everyone's,
funnelled by an inherited default.

The fix is a routing decision, not a retry, not a rate limiter, and not a change to how attempts
are scored.

## 1. The two runs

| | run 1 (16:13:54Z) | run 2 (16:29:54Z) |
|---|---|---|
| result | 7/8 pass^5 | 7/8 pass^5 |
| flaky check | F4 *remaining filesystem tools* — 3/5 | F27 *reasoning is a count* — 4/5 |
| model calls | 62 | 65 |
| 429s | 2 | 1 |

**The flake moved between runs.** F4-filesystem was 3/5 in run 1 and 5/5 in run 2. It is not a
property of any check — it is upstream, sampled. The model called `write_file` on every attempt
that got a response; the predicted failure mode (a model declining to call a tool) never occurred.

## 2. The retry never happened, and that is by the router's instruction

`ModelConfig.max_retries` is 2 and reaches the client — `build_model(...).root_client.max_retries`
is 2, verified. `BaseClient._should_retry` retries 429 by default. So the retries should have
absorbed this.

They never ran. From run 1's mirror, `chat_model_start` → `llm_error`:

| | start | error | elapsed |
|---|---|---|---|
| 429 #1 | 16:14:38.073 | 16:14:38.254 | **181 ms** |
| 429 #2 | 16:14:39.546 | 16:14:39.709 | **163 ms** |

`INITIAL_RETRY_DELAY` is 0.5 s, so two retries cannot cost under ~1.5 s of sleeping. 180 ms is a
single round trip.

### Measured offline, no router and no tokens

A local `BaseHTTPRequestHandler` returning 429 with chosen headers, with our own `build_model`
pointed at it:

| response | requests made | elapsed |
|---|---|---|
| plain 429 | **3** | 1.49 s |
| 429 + `x-should-retry: false` | **1** | 0.00 s |
| 429 + `Retry-After: 300` | **1** | 0.00 s |
| 429 + `Retry-After: 2` | **3** | 4.01 s |

Three requests over 1.49 s is `max_retries=2` with 0.5 s + 1.0 s backoff — **our configuration
works exactly as intended.** The live shape is the vetoed shape.

`_should_retry` reads `x-should-retry` and `Retry-After` *before* the status code and returns
`False` on either.

## 3. Which veto — answered

`mirror._retry_advice` was added after run 1 and captured it on run 2:

```json
{ "status_code": 429, "x-should-retry": "false" }
```

**The HF router sets `x-should-retry: false` on its `queue_exceeded` 429.** The openai SDK obeys by
design. Raising `max_retries` would do nothing at all.

This is the only change made to the harness during this investigation. It reads the headers off
`openai.APIStatusError.response`, so it costs no HTTP hook and no extra request.

## 4. What it is not

**Not our request rate.** Requests in the window before each 429:

| | prior 1 s | prior 5 s | prior 60 s |
|---|---|---|---|
| run 1, #1 | 2 | 4 | 25 |
| run 1, #2 | 1 | 6 | 27 |
| run 2 | **1** | **2** | 65 |

Run 2's 429 had one request in the prior second. That is not a burst, and **a client-side rate
limiter cannot pace its way out of another tenant's full queue.** An earlier draft of F46 blamed a
self-inflicted burst from the repeats; the rate data refutes it.

**Not an account quota.** Cumulative calls at the moment of the 429 were 25 and 27 in run 1, and 65
in run 2. No consistent threshold.

**Not `max_retries`.** Proven in §2.

**Not a scoring problem.** It was proposed that an infrastructure error should count as
*inconclusive* rather than failed, so pass^k would ignore it. Rejected, and the reasoning is worth
keeping: the harness owns `model.py` and the router connection, so making a model call succeed is
inside its remit. Relabelling the failure hides a real defect rather than fixing it. A model call
that fails at one request per second is a problem to solve, not a measurement to discard.

## 5. The mechanism

Eleven providers are live for `openai/gpt-oss-120b`, read from `/v1/models` on 2026-09-23:

| provider | first-token ms | throughput | $/M in | $/M out | context |
|---|---|---|---|---|---|
| cerebras | **166** | 1145 | 0.35 | 0.75 | 131072 |
| together | **220** | 91 | 0.15 | 0.60 | 131072 |
| baseten | 341 | 187 | 0.10 | 0.50 | 128072 |
| ovhcloud | 388 | 50 | 0.09 | 0.47 | 131072 |
| deepinfra | 396 | 31 | **0.037** | **0.17** | 131072 |
| fireworks-ai | 472 | 164 | 0.15 | 0.60 | 131072 |
| groq | 501 | **386** | 0.15 | 0.75 | 131072 |
| novita | 620 | 52 | 0.05 | 0.25 | 131072 |
| scaleway | 657 | 77 | 0.171 | 0.684 | *unstated* |
| nscale | 762 | 97 | 0.10 | 0.40 | 131072 |
| featherless-ai | — | — | — | — | *unstated* |

HF documents the model-id suffix as selecting a policy: `:fastest`, `:cheapest`, `:preferred`
(your configured order), or a named provider. **Omitting the suffix is equivalent to `:fastest`.**
`ModelConfig.model` defaults to a bare `openai/gpt-oss-120b`, so we are on `:fastest` by default,
never having decided to be.

Supporting evidence: exactly **two** distinct `system_fingerprint` values appear across each run of
60+ calls, interleaved — 35/20 in run 1, and 35/24 with the majority flipped in run 2. The router
is load-balancing between a small number of backends mid-run, consistent with a latency-ranked
policy over a contended head of the list.

*Inferred, not proven:* that those two fingerprints are the two lowest-latency providers. §8 says
how to settle it.

## 6. Blast radius

3 of 127 model calls across both runs returned 429 — **about 2.4%**.

```
P(a 62-call run sees zero 429s) ≈ 23%
→ roughly 3 runs in 4 go red on upstream capacity, with no regression present
```

**Caveat: n=3.** The error bars on 2.4% are wide and both runs were minutes apart on one afternoon.
The direction is what matters — pass^k over ~62 calls is fragile to a low-single-digit
infrastructure failure rate, and that holds at 1% too. A weekly cron that is red most weeks for
reasons that are not regressions is a cron people stop reading.

## 7. Options, with the constraints already verified

**Pin a provider.** `MODEL_ID` already accepts the suffix, so this is configuration, not code. It
also delivers two things `CLAUDE.md` already wanted for other reasons (F25): reproducibility, and a
context window that is a number rather than a range. Three motivations, one change. The cost is
losing whatever failover the router does on our behalf — though at a measured 2.4% 429 rate, that
failover is not currently earning much.

**`.with_fallbacks()` — ruled out.** It returns `RunnableWithFallbacks`, not a `BaseChatModel`:

```
build_agent        → CheckFailed: expected a BaseChatModel, got RunnableWithFallbacks
create_deep_agent  → AttributeError: 'ChatOpenAI' object has no attribute 'partition'
```

**`ModelFallbackMiddleware` — works, but is a partial ceiling.** Signature is
`(first_model, *additional_models)`. It reaches the **parent graph only**, because deepagents
inherits caller middleware into `general-purpose` only when the name shadows a default slot (F30) —
the same trap as `step_limit` before F24 and F39. Two further traps: pass **instances, never
strings**, because a model string re-activates deepagents' builtin `openai` provider profile and
its `use_responses_api=True`, which is F1 reversed (F28); and `init_chat_model` cannot parse
`org/model:provider` router ids anyway.

**A `BaseChatModel` fallback wrapper.** The only fallback that reaches every graph, since the model
object is what `resolve_model(spec["model"])` hands to each subagent. Real code, real tests, real
YAGNI cost.

**A custom `http_client` — recommended against.** It would work, by discarding the header the SDK
honours. The server said do not retry; overriding that on a shared inference router converts queue
pressure into worse queue pressure. It must also be passed at construction, since
`model_copy(update={"http_client": ...})` is silently ignored (F30).

## 8. On Jev / complexity-based routing

`jev-latest` is real: a TypeSafe AI "System One" model reached through `typesafe_sdk`
(`TypeSafeClient`, `Choice`), built for fast, calibrated, structured decisions against typed
questions. Routing prompts to a model by complexity is a genuine use for it.

**It does not address this failure.** Complexity routing optimises cost and quality; the measured
problem is availability. Against this harness specifically it would also add a dependency, a second
API key, and a network-capable path into a system whose prompt-injection argument currently rests
on having none (`CLAUDE.md`, *Prompt-injection threat model*) — plus a model call before every
model call, which is one more thing that can return 429.

**When it would earn its place:** once a domain exists with a real spread of task difficulty, so
there is a cost curve to optimise and a workload to measure against. Today every call is a fixed
smoke check, and there is nothing to route.

## 8a. Experiment: pinning a provider, 2026-09-23

Run with `MODEL_ID=openai/gpt-oss-120b:groq`, `LIVE_CHECK_REPEATS=5`. Two runs.

| log | route | calls | 429s |
|---|---|---|---|
| `20260923T161354Z-81b4f402` | unpinned (`:fastest`) | 62 | **2** |
| `20260923T162954Z-eed4d965` | unpinned (`:fastest`) | 65 | **1** |
| `20260923T192809Z-ff9b2a2f` | `:groq` | 65 | **0** |
| `20260923T193356Z-394514a7` | `:groq` | 65 | **0** |

The second pinned run came back **8/8 pass^5, exit 0**:

```
model:  openai/gpt-oss-120b:groq
repeats: 5 per check (exit code reads pass^5)
  [5/5] F1   [5/5] F2   [5/5] F4   [5/5] F4
  [5/5] F5   [5/5] F26  [5/5] F27  [5/5] F25

8/8 checks passed (pass^5)
```

**Strength of the evidence.** 0 in 130 pinned calls, against a measured unpinned rate of 2.36%:

```
P(0 in 130 calls | 2.36%) = 4.5%   → about 1 in 22 by chance
```

That is supporting, not proof — a single afternoon, one pinned provider, no control for the router
simply being quieter at 19:28 than at 16:13. What lifts it is that the mechanism in §5 predicts
exactly this, and the fingerprint decode below confirms the predicted culprit was in the mix.

### The fingerprint decode

One tiny call per pinned provider:

| provider | fingerprint |
|---|---|
| **cerebras** | **`fp_752b9cb17e04d95d05c3`** |
| groq | `fp_c3da0d4bb9` |
| featherless-ai | `fp1-nst-nes` |
| together | `default` |
| baseten, ovhcloud, deepinfra, scaleway, nscale | *(none emitted)* |
| fireworks-ai | 402 — pay-as-you-go not enabled for this account |

**`fp_752b9cb17e04d95d05c3` is cerebras**, and it is one of the two fingerprints serving both
unpinned runs. Cerebras advertises the lowest first-token latency of all eleven providers (166 ms),
which is exactly what `:fastest` would select. The second fingerprint,
`fp_b546658c8e93d2e57ef2`, is still unidentified — none of the probed providers returned it, and
five emit no fingerprint at all.

### A bug the experiment found

The first pinned run aborted:

```
CheckFailed: the router does not list openai/gpt-oss-120b:groq; the check has no subject
```

`check_every_provider_serves_the_context_we_assume` looked `config.model` up in `/v1/models`, which
lists the **bare repo id**. So pinning a provider — F25's own recommended mitigation — broke the
check that records F25. Fixed: `_model_route()` splits the suffix, a policy suffix
(`:fastest`, `:cheapest`, `:preferred`) still means "assert across all providers", and a named
provider narrows the check to that one. An unserved pin is refused rather than narrowed to an empty
set, which would otherwise pass by measuring nothing. Four mutants killed.

## 9. Recommended order

1. ~~Decode the fingerprints.~~ **Done** — §8a. Cerebras identified; one fingerprint still unknown.
2. ~~Pin a provider and re-run.~~ **Done** — §8a. Zero 429s in one run, and a real bug fixed on the
   way.
3. **Decide whether to pin by default.** `ModelConfig.DEFAULT_MODEL` is still the bare id, so the
   default is still `:fastest`. Changing it is a one-literal change plus its pin test, and it
   should be a deliberate decision: it buys routing determinism, reproducibility and a known
   context window (F25), and it costs whatever failover the router does on our behalf.
4. **Widen the error capture.** `mirror._retry_advice` records three headers; recording all of them
   would let a future 429 name its own provider. Free, and still unfinished.
5. **Only if 429s return under a pin**, revisit fallback, knowing from §7 that anything covering
   subagents means a `BaseChatModel` wrapper.

None of this needs a new dependency or a change to the harness's architecture.

## 10. Open

- **Which providers the two fingerprints are.** Settled by step 1.
- **Whether pinning removes the 429s.** Unmeasured. It is a hypothesis with a mechanism, not a
  result.
- **What the router's actual per-provider limit is.** Not published as far as this looked, and not
  measured.
- **Whether 2.4% holds.** n=3 on one afternoon.
- **Whether `:preferred` behaves differently from a hard pin**, and whether either does server-side
  failover on a 429. HF documents automatic failover for `provider="auto"` in `huggingface_hub`; it
  is not established that the OpenAI-compatible router applies the same behaviour to a policy
  suffix.

## Reproducing any of this

```bash
# Rate and cumulative-count analysis around each 429, from a mirror log.
# Fingerprint distribution per run.
# Both are short json/datetime scripts over logs/*.jsonl — `event` is
# "chat_model_start" / "llm_end" / "llm_error"; provider identity is
# metadata.system_fingerprint on llm_end; retry_advice is on llm_error.

# The offline retry probe (§2): a BaseHTTPRequestHandler returning 429 with
# chosen headers, with build_model pointed at 127.0.0.1. No tokens, no router.

# The live provider table (§5): one catalogue GET, no inference.
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
from my_agent.model import ModelConfig
from my_agent.main import _router_models
cfg = ModelConfig.from_env()
for m in _router_models(cfg):
    if m.get('id') == 'openai/gpt-oss-120b':
        for p in m.get('providers', []):
            print(p)
"
```
