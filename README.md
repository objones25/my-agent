# my-agent

A deep-agent harness built on [`deepagents`](https://github.com/langchain-ai/deepagents), driven by
contracts, tests and observability rather than by feature count.

**The agent's domain — the job it does and the tools that job needs — is deliberately
undecided.** What is being built here is the harness: the capability allowlist, the model wiring,
the run bounds, the observability seams, and the tests that keep all of it honest. Picking a domain
first tends to produce a harness shaped around one task's accidents; this order tests the seams
against a domain that does not exist yet, which is the harder case. No domain tool is bound today:
the agent runs on deepagents' own filesystem tools plus `task`, which is enough to exercise every
seam. When a real domain is chosen, it should slot in behind the protocols listed in CLAUDE.md
(*Architecture: protocol-driven contracts*) without any existing file changing.

## What is actually interesting here

- **A capability allowlist that is proved, not requested.** `create_deep_agent` enables a shell
  `execute` tool with no opt-in. It is withheld, and the absence is asserted against the compiled
  graph *and* every subagent graph — because deepagents gives its general-purpose subagent its own
  filesystem middleware, so the parent's allowlist is not the whole story.
- **Every bound lives next to what it constrains, not at the call site.** A turn goes through
  `run_turn`, which carries four no caller can forget — a step limit (25), a wall clock (600s), a
  token budget (500k) and a cap on how many times a paused turn may resume (3). Exceeding one
  raises `StepLimitExceeded`, `DeadlineExceeded`, `TokenLimitExceeded` or `ResumeLimitExceeded`:
  handled errors the CLI reports, never a traceback. Two further caps — on tool calls and on
  `task` dispatches — are installed by `build_agent` and apply to the whole agent. They exist
  because a *step* is not a *call*: langgraph runs every tool call in one `AIMessage`, so a model
  that fans out ten calls does ten times the work while the step limit sees one step. Those two
  block the offending call and let the agent answer with what it has.
- **A turn returns a record, not just a reply.** `TurnResult` carries the tool calls that failed,
  whether an answer was ever begun, and the agent's own filesystem as the graph returned it, so
  "did it do the thing" has something to read that is not the model's prose.
- **Library behaviour recorded rather than assumed.** `docs/findings.md` holds forty-seven
  verified findings, each with how it was checked and what the code does about it. Eight of them
  are additionally re-checked against the live router by `uv run my-agent`.
- **Tests that are checked for being able to fail.** Changes here are verified by breaking the
  code a test covers and confirming the test goes red. When one stays green against that
  deliberate break, the blind spot is written down rather than quietly fixed (F39, F42).

## Requirements

- Python 3.13 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/)
- A Hugging Face token with Inference Providers access

## Setup

```bash
uv sync
cp .env.example .env         # then fill in HF_TOKEN
```

`HF_TOKEN` is the only environment variable you must set. Tracing switches on if you also supply
the LangSmith or W&B keys; see `.env.example`.

Optionally, to run the offline gate on every commit:

```bash
uv run pre-commit install
```

The hook is configured but never installed for you -- and re-running this in a fresh clone is on
you. See CLAUDE.md, *The gate*.

## Running it

```bash
uv run my-agent                  # the eight live checks against the real router
uv run my-agent "your prompt"    # one ordinary turn
```

Both spend real tokens. Each check runs `LIVE_CHECK_REPEATS` times (5 by default; set the
environment variable to override). The run prints a header, then one block per check, then a tally:

```text
model:  openai/gpt-oss-120b
tracing: langsmith, weave
log:    logs/<timestamp>-<id>.jsonl
tools:  ['delete', 'edit_file', 'glob', 'grep', 'ls', 'read_file', 'write_file'] (+ task)

repeats: 5 per check (exit code reads pass^5)

  [5/5] F1  chat-completions endpoint reachable
         reply='pong'
  ...  (seven more, one per check)

8/8 checks passed (pass^5)
```

**Exit status is 0 only if every check passed every attempt** — pass^k, not pass@k, because these
are invariants rather than best-effort tasks. A check that passed four times in five is reported as
flaky and still fails the run. The eight checks and their recorded evidence are tabulated in
`docs/findings.md`, *Live verification*.

## Development

```bash
./scripts/check.sh               # every offline gate, in order -- what CI runs
uv run pytest                    # the offline suite on its own
uv run pytest -m live > live.log 2>&1   # hits the real router; redirect, never pipe
```

`scripts/check.sh` is the single source of truth for the gate: the pre-commit hook (once installed,
see Setup) and `.github/workflows/ci.yml` both call it and re-list nothing.

## Where to look

| Path | What it holds |
|---|---|
| `CLAUDE.md` | The contributor's contract: architecture, non-negotiables, verified API facts |
| `docs/findings.md` | F1-F47: verified library, provider and test-suite behaviour, and what the code does about each |
| `evals/` | Empty. Where a model-dependent eval suite would go once a domain exists |
| `docs/superpowers/specs/` | Design documents |
| `src/my_agent/capabilities.py` | The allowlist and the proof it held |
| `src/my_agent/run.py` | One bounded turn |

## Status

Pre-domain. The harness works end to end against the Hugging Face router, with LangSmith and W&B
Weave tracing verified to coexist over the same run. `evals/` is empty: the eight live checks are
the only end-to-end measurement. They now run five times each and are scored pass^k, so a flake
and a regression no longer look alike (F45) — but nothing grades the model's prose, which is a
decision (see CLAUDE.md, *What is deliberately not verified*) rather than an omission.
