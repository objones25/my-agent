# my-agent

A deep-agent harness built on [`deepagents`](https://github.com/langchain-ai/deepagents), driven by
contracts, tests and observability rather than by feature count.

**The agent's domain is deliberately undecided.** What is being built here is the harness: the
capability allowlist, the model wiring, the run bounds, the observability seams, and the tests that
keep all of it honest. No domain tool is bound today: the agent runs on deepagents' own filesystem
tools plus `task`, which is enough to exercise every seam. When a real domain is chosen, it should
slot in behind the existing protocols without any existing file changing.

## What is actually interesting here

- **A capability allowlist that is proved, not requested.** `create_deep_agent` enables a shell
  `execute` tool with no opt-in. It is withheld, and the absence is asserted against the compiled
  graph *and* every subagent graph — because deepagents gives its general-purpose subagent its own
  filesystem middleware, so the parent's allowlist is not the whole story.
- **Bounds that belong to the thing they bound.** One turn goes through `run_turn`, which always
  sends a step limit and attaches a wall-clock deadline, so no caller can forget either.
- **Library behaviour recorded rather than assumed.** `docs/findings.md` holds forty-four
  verified findings, each with how it was checked and what the code does about it.
- **Tests that are checked for being able to fail.** Changes here are verified by reverting the fix
  and watching a test go red — and when one survives its mutant, that is recorded too (F39, F42).

## Requirements

- Python 3.13 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/)
- A Hugging Face token with Inference Providers access

## Setup

```bash
uv sync
cp .env.example .env         # then fill in HF_TOKEN
uv run pre-commit install    # optional: run the offline gate on every commit
```

The hook is configured but not installed by default -- see CLAUDE.md for why.

Only `HF_TOKEN` is required. Tracing switches on if you supply the LangSmith or W&B keys; see
`.env.example`.

## Running it

```bash
uv run my-agent                  # live checks: one per finding in docs/findings.md
uv run my-agent "your prompt"    # one ordinary turn
```

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
| `docs/findings.md` | F1-F44: verified library behaviour and what the code does about each |
| `docs/superpowers/specs/` | Design documents |
| `src/my_agent/capabilities.py` | The allowlist and the proof it held |
| `src/my_agent/run.py` | One bounded turn |

## Status

Pre-domain. The harness works end to end against the Hugging Face router, with LangSmith and W&B
Weave tracing verified to coexist over the same run.
