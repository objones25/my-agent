# CLAUDE.md

Guidance for Claude Code in this repository. `README.md` says what the project is; this is the
contributor's contract, and **`docs/findings.md` (F1–F47) is the evidence behind it** — read it
before debugging anything that looks like a library bug, and add to it when you verify something
new.

A deep-agent harness on `deepagents`, driven by TDD, evals and observability rather than by feature
count. The agent's **domain is deliberately undecided** (see "Scope discipline"); what is being
built is the harness: contracts, model wiring, the capability allowlist, run bounds, observability
seams, and the tests that keep them honest. Model access goes through the Hugging Face Inference
Providers router via `langchain-openai`; observability is LangSmith and W&B Weave, each behind its
own narrow protocol so neither is load-bearing.

## Commands

```bash
./scripts/check.sh                        # every offline gate, in order. What CI and the
                                          # pre-commit hook both run; the list lives only here.
uv sync                                   # install/refresh the locked environment
uv run pytest                             # unit tests (live + eval cases deselected)
uv run pytest -m live > live.log 2>&1     # real HF router / LangSmith / W&B. Redirect, never
                                          # pipe (`| tail; echo $?` yields tail's status), and
                                          # expect ~5 min to exit after a ~4s suite (F23).
uv run pytest -m eval                     # the eval suite
uv run my-agent                           # the eight live checks (docs/findings.md, "Live
                                          # verification") x LIVE_CHECK_REPEATS, scored pass^k
uv run my-agent "your prompt here"        # one ordinary turn instead
uv run ruff check . --fix                 # lint
uv run mypy                               # type check (strict; src + tests)
uv run pyright                            # second type checker (standard; see F16)
# The negative-space audit alone. check.sh also runs `--select NSP001 --min-assertions 2`, advisory.
uv run python scripts/audit_negative_space.py src/ --select NSP002,NSP003,NSP005,NSP006,NSP007
```

## Non-negotiables

**Never write an API call from memory.** These libraries move faster than any training cutoff, and
this project pins recent majors (`langchain` 1.x, `openai` 3.x, `pytest` 9.x, `mypy` 2.x) where most
published examples are a major behind. Before writing code against any library:

1. `npx ctx7@latest library "<Official Name>" "<the question>"` → pick the `/org/project` ID, then
   `npx ctx7@latest docs <id> "<the question>"`.
2. **Then confirm against the installed code**, because docs lag the wheel:
   ```bash
   uv run python -c "import inspect, X; print(inspect.signature(X.thing))"
   uv run python -c "import X; print(sorted(n for n in dir(X) if not n.startswith('_')))"
   uv run python -c "import X; print(X.Model.model_fields.keys())"   # pydantic models
   ```

ctx7 gives the intent, `inspect` the truth. When they disagree `inspect` wins, and the disagreement
is worth a finding.

## Least privilege

**A capability the agent has not been given cannot be misused.** Every tool, permission, filesystem
path and network reach is off unless something needs it, and turning one on is a deliberate,
reviewed edit — never inherited from a library default. Not theoretical: `create_deep_agent` enables
a shell `execute` tool with no opt-in (F4); `capabilities.py` withholds it.

- `DEFAULT_FILESYSTEM_TOOLS` is defined by *subtraction*, and a load-time check asserts it equals
  `get_args(FsToolName) - {"execute"}`. A new upstream tool fails the import rather than being
  granted silently.
- `AgentConfig.middleware` and `.permissions` default to empty. Empty permissions means "no rules",
  not "deny all" — if the threat model needs denial, write them.
- **An allowlist is only a request until it is checked.** Postconditions assert the withheld tool is
  absent from the compiled graph **and** that the graph bound *some* tools, so the absence cannot
  pass vacuously.
- **The parent graph is not the whole allowlist.** `create_deep_agent` auto-adds a general-purpose
  subagent behind `task` with its own `FilesystemMiddleware` and no `tools=` allowlist, so
  `_require_shell_withheld` reads every subagent graph back via `capabilities.subagent_graphs` and
  asserts there too. `compiled_tools` is public because the permission rules reach that graph by
  *object identity*, which is the part worth asserting (F20).
- **A pin on a parameter list does not cover a door that is not a parameter.** deepagents resolves
  a `HarnessProfile` keyed by the model's provider and id from a process-global registry, populated
  by running a callable from every installed distribution advertising a `deepagents.harness_profiles`
  or `deepagents.provider_profiles` entry point. Such a profile adds middleware, drops tools,
  rewrites tool descriptions and rewrites the system prompt — none of it through
  `create_deep_agent`. `capabilities.DEEPAGENTS_PLUGIN_GROUPS` asserts both groups are empty at
  import. **And an entry point is not the only door that is not a parameter.** deepagents also
  `import_module`-probes `langchain_aws` and `langchain_fireworks` and appends their prompt-caching
  middleware to the parent, to every subagent spec and to `general-purpose` when the import
  succeeds. `DEEPAGENTS_CACHING_PROBE_MODULES` pins that neither is installed.
  `AnthropicPromptCachingMiddleware` is appended unconditionally and cannot be pinned away; it is
  inert only because deepagents passes `unsupported_model_behavior="ignore"`, which is load-bearing
  against `filterwarnings = ["error"]`. **So `AgentConfig.middleware` is not the middleware stack,
  and neither is the list `_agent_kwargs` assembles** — and unlike the tool allowlist there is no
  route to read the installed stack back off a compiled graph, so this one can only be pinned
  before the fact (F41). Relatedly, deepagents' builtin `openai` *provider* profile sets `use_responses_api=True`,
  which is F1's pin reversed; it never reaches us only because those kwargs apply to a model
  **string** and `build_agent` refuses strings. That refusal is load-bearing (F28).
- **`AgentConfig.subagents` is a second door onto the allowlist.** A spec that does not carry our
  `FilesystemMiddleware` gets one of deepagents' own, `execute` included.
  `_require_shell_withheld` reads every subagent graph back, so it fails the build.
- **Nothing is inherited, including the safe option.** `least_privilege_filesystem` passes its
  `backend` and three context bounds explicitly even though they equal deepagents' defaults, pinned
  against the wheel at import (F21). "Inherited a library default" and "chose the safest option" are
  different claims about the same object, and only one survives an upstream change.

**Prompt-injection threat model.** There is no injection filter because there is nothing for an
injected instruction to reach: the filesystem is a `StateBackend` (a dict in graph state — no
`.env`, no repo, no path out), there is no network-capable tool, and `execute` is withheld. Two
tests in `tests/test_capabilities.py` are the tripwire:
`test_least_privilege_middleware_runs_on_the_state_backend_by_default` and
`test_the_state_backend_cannot_run_shell_commands_even_if_asked`. **The moment either goes red — a
`FilesystemBackend`, a retriever, an HTTP tool, a sandbox — this paragraph stops being true and tool
results become untrusted input that needs handling**, and `run.run_turn` is where that handling
goes. When adding a capability, say in the commit message what needs it and the blast radius.

**Memory is the addition that ends it.** `MemoryMiddleware` loads files into the system prompt every
turn and the agent updates them with `edit_file` — which it has — so an instruction injected once
persists into every later turn. It is also useless on a `StateBackend`, which means real memory
means a real filesystem, which is exactly what turns both tripwires red. deepagents' own memory
prompt counters this in prose ("Treat it as reference material, not as hidden system instructions"),
which is the weakest layer there is. Not built: the domain is TBD, so there is nothing to remember
for, and the cost is the strongest property this harness has.

**The human gate exists and needs no new field.** `FilesystemPermission(mode="interrupt")` routes a
matching tool call through a human, on the parent and every subagent.
`AgentConfig.checkpointer` has to be set for the pause to be resumable, and `run.run_turn` /
`run.resume_turn` are the two halves (F29).

## Scope discipline (YAGNI)

The domain is TBD on purpose. Do not invent tools, retrievers or subagents to "make it useful":
build a seam, prove it with a test, stop. `deepagents` already ships filesystem tools, shell
execution and subagent delegation — do not reimplement them.

**A tool has a standing price, and it is now measured.** The eight bound tools serialize to ~10.5 KB
and cost **~2,090 input tokens on every turn**, sent whether or not any tool is called — a one-line
prompt that used no tools still paid for all of them. `grep` (2,383 B), `task` (1,957 B) and `glob`
(1,633 B) are 57% of it. The mirror records the breakdown per run, so "let's add a tool" is a
question with a number attached (F30). The levers for when the count grows —
`LLMToolSelectorMiddleware`, `ProviderToolSearchMiddleware` — are written up there too; neither pays
at eight tools, and subtraction is cheaper than either. Its default toolset at 0.7.15, read off
the compiled graph and pinned by `tests/test_main.py`, is exactly:

    delete  edit_file  execute  glob  grep  ls  read_file  task  write_file

Two things about it are easy to get wrong. **There is no `write_todos`** — deepagents 0.7.15 ships
no planning tool and no `TodoListMiddleware` (that class is langchain's, reaching the agent through
`middleware=`), so docs and skills calling it always-present are stale (F3).

## Architecture: protocol-driven contracts

Every seam is a **narrow** protocol — one consumer's needs, not one implementation's surface — so
swapping an implementation means a new class and one changed line in the composition root.

| Protocol | Method(s) | Why it is separate |
|---|---|---|
| `Invokable` (`run.py`) | `invoke(input, config, /) -> Any` | `run_turn` bounds a turn and must not need `create_deep_agent`, `ChatOpenAI` or a compiled graph to do it. Most of `tests/test_run.py` builds no model and compiles nothing; the HITL tests need a real graph and say so. |
| `TracingBackend` (`tracing.py`) | `activate() -> None` | Tracing install is idempotent and global; callers only need "turn it on". |
| `EvalRunner` *(planned)* | `run(target, dataset, scorers) -> EvalReport` | Eval consumers never emit spans; tracing consumers never score, so this is deliberately **not** folded into `TracingBackend`. |

- Use `typing.Protocol` for seams this project defines — structural typing keeps adapters decoupled
  — and extend by adding a new protocol, not by widening an existing one.
- `Protocol` enforces nothing at instantiation: a class inheriting one and implementing nothing
  instantiates fine and the missing method returns `None`. Where that silence would be a production
  incident, use `abc.ABC`. `@runtime_checkable` checks *presence* only, and is slow.
- A protocol variable must be declared `ClassVar` and implemented as one, **or** declared a
  read-only `@property` and implemented as an instance attribute; the other two combinations fail
  mypy or pyright (F16). Run any new non-method protocol member past both checkers.

### Configs are parameter objects, not argument lists

A factory takes **one config object**, never a growing list of keyword parameters. Each config's
field names are exactly the callee's parameter names, and the factory splats them:

```python
model = ChatOpenAI(**config.as_kwargs(), use_responses_api=USE_RESPONSES_API)
agent = create_deep_agent(model=model, **agent_config.as_kwargs())
```

Adding a setting is then one new field with a default — `build_agent` does not change and no call
site changes. `contracts.check_config_contract` makes the splat safe by verifying every field
against the callee's real parameters **at import time**, so a typo or an upstream rename fails on
import by name instead of becoming a `TypeError` from inside the library. Two pins cover what it
cannot see: `KNOWN_CREATE_DEEP_AGENT_PARAMS` (a *new* upstream parameter appearing — exactly how
`execute` arrived switched on), `KNOWN_OUTPUT_STATE_KEYS` (a new *output* key appearing — exactly
how `files` came back on every turn and was read by nothing, F43) and `ModelConfig._ENV_FIELDS` checked against `dataclasses.fields()`
(a field becoming unreachable from the environment). Full reasoning in F19.

**Composition root.** Concrete classes are chosen in exactly one place — `main.py`. Nothing below it
*decides which backend to use* by inspecting `os.environ`. A `*.from_env(env=None)` classmethod may
read `os.environ` as its own documented default, the way an injectable default argument reads
anything else: the caller can always override it with an explicit mapping. **If a test has to set an
env var to reach the line under test, the wiring is in the wrong place.**

## Negative space programming

Bugs live in the states the code was never written to handle. Write those down as executable checks.

- `negative_space.py` holds `require()`, `unreachable()` and `bounded()`. Use these, not bare
  `assert` — `python -O` deletes `assert` statements entirely, condition and message both, and some
  container images set `PYTHONOPTIMIZE`. Write preconditions before the body and postconditions
  after it, asserting the positive space *and* the negative space: not just "k is in range" but "the
  two halves do not overlap".
- **`require()` for programmer errors** (a caller you own passed something impossible) — crash.
  **`raise ValueError`/typed exceptions for operating errors** (env var missing, router 503,
  malformed model output) — handle at the edge. Same predicate, different category, decided by where
  the value came from. Model output is *always* an operating error: it is untrusted input.
- Every loop, retry and agent turn gets an explicit bound — an agent that loops forever is the worst
  failure mode here. What bounds a turn is `run.RunBounds`: `step_limit` (25), sent as
  `recursion_limit` on every `run_turn` config, the wall-clock `deadline_s` (600) enforced by
  `RunDeadline`, `token_limit` (500,000) enforced by `RunTokenBudget`, and `resume_limit` (3). None
  goes through `bounded()`, which **has no call site in `src/` at all** — it exists for iteration
  loops this codebase does not have yet.
- **A turn is billed in none of the units the first three bounds count.** Steps, seconds and halves
  are all satisfiable by a turn that spends millions of tokens: 37 model calls at a 96,000-token
  conversation is ~3.5M tokens inside every other bound. `RunTokenBudget` is `RunDeadline` with a
  different unit — accumulate on `on_llm_end`, refuse on `on_chat_model_start` — reading the same
  `usage_metadata` the mirror records, and it reaches subagent calls, which is where the tokens go.
  `unmeasured_calls` is counted separately, because a provider that omits usage makes the bound
  blind and a blind bound must not be silent (F35).
- **A bound applied per invocation is not a bound on a turn, and a pause splits a turn into
  invocations.** Both of the first two used to reset on every `resume_turn`, so a turn approved ten
  times got ten full budgets. `deadline_s` now carries: `TurnResult.elapsed_s` accumulates the
  agent's own wall clock and a resume runs on the remainder (a human's deliberation is never charged
  — the clock is read at invocation boundaries). `step_limit` **cannot** carry: langgraph restarts
  the superstep count and never reports what it reached, so `resume_limit` counts the halves
  instead. Worst case is four step budgets, not unboundedly many (F33).
- **A bound sent is not a bound applied, and the graph you configure is not the only graph that
  runs.** `step_limit` reaches the parent and stops there: langchain's `create_agent` binds
  `recursion_limit: 9999` onto every graph it compiles, and deepagents invokes a subagent with that
  graph's own bound config, which wins. Measured: `step_limit=25` allowed 12 parent model calls
  alone and **5002** once each step dispatched a `task` subagent. `capabilities.SUBAGENT_STEP_LIMIT`
  is the second bound, applied by compiling the agent, reading deepagents' own subagent back out and
  handing it back rebound — and asserted by reading it off the graph afterwards (F24). **"Every
  graph" includes the parent.** It carried 9999 too, so anything not going through `run_turn` ran
  3,325 model calls before langgraph stopped it. `capabilities.PARENT_STEP_LIMIT` is bound onto the
  returned graph; `run_turn`'s per-invocation limit still wins, because at the top level an explicit
  invoke config beats the graph's own bound config — the opposite of the subagent case (F39).
- All four bounds fail the same way at the edge, as operating errors `main` reports rather than
  crashes: `DeadlineExceeded`, `StepLimitExceeded` (which translates langgraph's
  `GraphRecursionError` — before it existed the wall clock was a handled ceiling and the step count
  was a traceback), `TokenLimitExceeded` and `ResumeLimitExceeded`. All four subclass
  `run.BoundExceeded`, which is the one name `main` catches, so a fifth bound is reported without
  editing it.
- **A step is not a call.** langgraph's tool node runs every call in one `AIMessage`, so a fan-out
  does ten times the work per step and `step_limit` sees one step either way.
  `capabilities.call_limits()` installs the two bounds that can see it: `TOOL_CALL_LIMIT` (24,
  all tools) and `TASK_DISPATCH_LIMIT` (3, `task` only — `SUBAGENT_STEP_LIMIT` bounds how far one
  dispatch runs, this bounds how many there are). Both `exit_behavior="continue"`: the exceeded
  call is blocked and the agent answers with what it has, which beats crashing a turn already
  bounded twice over. Installed by `build_agent`, not offered through `AgentConfig` — a bound a
  caller has to remember is a bound that will be forgotten.
- **`AgentConfig.middleware` reaches the parent only.** deepagents inherits caller middleware into
  the general-purpose subagent only when its `.name` shadows a default slot, which is why our
  `FilesystemMiddleware` gets there and a call limit does not. Anything relied on as a *global*
  ceiling must be checked on both graphs (F30).
- **A bound belongs to the thing it bounds, not to the call site.** `RECURSION_LIMIT` lived in
  `main.py`, so every other caller of `build_agent` inherited langchain-core's default by accident —
  and that default is *also* 25, which is what made it look like a decision. A postcondition must be
  relative for the same reason: `len(messages) > 1` passes on any non-empty history while the agent
  contributes nothing, so the check is `> len(sent)`.
- **Some bounds cannot be set from code, and saying so is the deliverable.** `weave.init()` signs
  the process up for an `atexit` flush that burns a hardcoded 300s. There is no setting for it; two
  plausible levers were tried, measured and reverted (F23).
- Split compound checks: `require(a); require(b)` names the failure, `require(a and b)` does not.
  And mypy cannot narrow through `require()`, so where a check also narrows (`x is not None`) use an
  explicit `if ... raise CheckFailed(...)` — same behaviour at runtime, and mypy follows it (F10).
- Plain `assert` stays correct in test bodies (pytest rewrites it for readable failures), but **a
  test expecting a tripped `require()` names `CheckFailed`, never `AssertionError`.** The latter is
  its base class, so an `AssertionError` expectation is also satisfied by a bare `assert` — silently
  accepting a `require()` downgraded to the one construct this project bans, which `python -O` then
  deletes entirely. Measured, not assumed.

## Testing and evals

Keep them apart. 526 offline tests and 2 live as of 2026-09-23; `evals/` is still empty.

- **Unit tests** (`tests/`, default selection) are deterministic and offline. One test file per
  source module; a new module gets a new file, not an extra section in an existing one. They test
  the harness: protocol conformance, wiring, bounds, error paths. For each `require()`, a test that
  trips it — that is what turns a contract into a tested contract. **Outside one named exclusion,
  that is now the state rather than the goal.** Measured 2026-09-23 by wrapping `require()` and
  `CheckFailed` in pytest plugins that log their call site whenever one raises, then diffing against
  an AST walk — not by reading coverage, and not by counting by hand: **111 `require()` sites, 101
  tripped; 16 `raise CheckFailed` sites outside the helpers, 15 tripped.** Re-measured with a
  `sys.monitoring` RAISE plugin and an AST walk; it counts one fewer `require()` site on the tree the
  earlier figure came from, and the old tool is not in the repo to say why.

  **All 11 that remain are in `main.py`**, left alone deliberately — it is scaffolding, and a check
  there is worth a findings entry rather than a fix round. **No import-time check is untripped any
  more**: the ones a reload can reach are driven by the `tripping_an_import_time_check` fixture, and
  the ones it could not reach moved into functions (F44, and the bullet below).

  Nothing else in `src/` has an untripped check. **Keep it that way**: a new `require()` lands with
  the test that trips it, or the count above stops being true and nobody notices, which is the
  failure this project spent a branch measuring. Three techniques cover all of it — a bad
  argument for a precondition; `monkeypatch` on the name the module actually calls for a read-back
  postcondition, building the real library object and then dropping one setting
  (`_forgetful_filesystem` in `tests/test_capabilities.py` is the pattern); and, for a check that
  runs at *import*, the `tripping_an_import_time_check` fixture, which patches the library and
  re-imports.

- **A load-time check belongs in a function, not at module level.** Module-level is only drivable by
  `importlib.reload`, and a reload **rebinds every class the module defines** — so
  `isinstance(cfg, AgentConfig)` inside the reloaded module fails against an instance any other test
  module is holding, and reads `expected an AgentConfig, got AgentConfig`. Measured: 15 unrelated
  tests went red that way, **none of them when run alone** (F44). The fixture now refuses to reload
  a module that defines classes, and `contracts.check_known_parameters` /
  `check_required_parameters` / `capabilities.require_compaction_fits_the_window` /
  `model.require_env_fields_cover_the_config` are the shape to copy: called at import, so the
  guarantee is unchanged, and callable by a test with bad arguments.

  Two traps met while doing it. **A `require()` message is evaluated whether or not the check
  fails** — `require(cond, f"...{read_back(x)}")` calls `read_back` every time, which is why
  `build_agent` calls `bound_step_limit` five times and not three; a test that stubs such a helper
  by call count must count, not assume. And **two sites with the same message cannot be told
  apart**: `RunDeadline`'s two backwards-clock checks were byte-identical, so `match="backwards"`
  matched either and the pair read as covered while only one had a test. Give every check a message
  no other check could produce.

  Line coverage hides all of it — `require()` is a
  function, so the
  raise lives in `negative_space.py` and the call site reads as covered whether or not the predicate
  ever went false.
- **Evals** (`evals/`, `-m eval`) measure model-dependent behaviour and are allowed to be
  non-deterministic and slow. A failing eval is a signal, not a broken build. Empty today — the
  eight live checks in `main.py` are the closest thing, and they are *not* evals: they assert
  harness invariants, so they are scored **pass^k** and a single failed attempt fails the run.
  **A non-deterministic check needs more than one attempt or it measures nothing you can act on.**
  `LIVE_CHECK_REPEATS` (5) is uniform across all eight on purpose: which of them can actually vary
  has not been measured, and calling five of them deterministic would be the untested assumption
  the repeats exist to remove. Pin per-check counts once there is a number (F45).
- `-m live` marks anything touching the HF router, LangSmith or W&B. `addopts` carries `-m "not live
  and not eval"`, so both are deselected by default; a command-line `-m live` overrides it.
  Registering a marker does *not* deselect it — a real gap here until it was measured.
- **"Offline" is enforced, not assumed.** An autouse `_forbid_network` fixture in
  `tests/conftest.py` fails any test in that directory that opens a socket, stepping aside only for
  `live`. Shared setup (`valid_key`, `valid_secret`, `deny_secrets`, `assert_does_not_raise`) lives
  there too as fixtures, so no test can leak a mutation into the next. It does **not** reach the
  `src/` doctests, which run in the same suite (`--doctest-modules`) from a separate `testpaths`
  entry — harmless today, but read `docs/findings.md`, "Test-infrastructure specifics", before
  changing either.
- **A passing suite is not a passing state if the tests cannot fail.** Before trusting new tests,
  break the code they cover and watch them go red; twenty-three such mutants are recorded in
  `docs/findings.md`, and a test that survives one is decorative — two did, and are recorded as
  findings in their own right (F39, F42). Two corollaries: **assert the
  claim and its discriminator** (`test_build_agent_withholds_the_shell_tool_from_every_subagent` is
  worthless without `test_a_bare_deep_agent_does_grant_the_shell_tool_to_its_subagent`, which keeps
  passing if deepagents stops granting `execute` for its own reasons), and **a pinned value equal to
  the library default cannot be tested by reading it back** — record the *call* instead (F21).
- **A constant compared to itself is not a pinned constant.** The wider form of F21's rule: if the
  expected value in a test can be changed by editing `src/`, the test is a tautology. Five constants
  moved at once — including the router base URL to `api.openai.com`, plus a 10x wall clock and a 10x
  token ceiling — and all 376 tests stayed green (F42). Every decision now has one literal pin.
  Write the number down, and pin the default *object* too: pinning `TOKEN_LIMIT` does not pin the
  `RunBounds` field that defaults to it.
- **A mutation-verified test is only verified against the mutant you chose.** A behavioural test for
  the parent step limit survived deleting the fix, because the fake it used dispatched a subagent
  and hit `SUBAGENT_STEP_LIMIT` first — the turn aborted early whatever the parent was bound to.
  Pick a fake whose only possible stopper is the bound under test (F39).
- Do not run the suite under `python -O`: it exits 1 rather than lying to you. `filterwarnings =
  ["error"]` is set, and a new deprecation warning from these fast-moving libraries fails the build
  on purpose — fix it, or scope an ignore matched on message *and* category *and* module as the one
  existing entry is (F11). Do not widen the setting.

### What is deliberately not verified: the model's output

**Nothing grades the model's prose, and that is a decision, not an oversight.** `run_turn` bounds
a turn and asserts that one happened; it trusts nothing about the *content* and checks nothing about
it either.

What it does carry is the *record*: `failed_tool_calls`, `answered` and `files` — the agent's
filesystem as the graph returned it. That last one was available on every turn from the start and
was being discarded (F40), which is worth stating plainly, because this section used to say the
ladder's first rung needed a domain. It did not; it needed a field. What still needs a domain is the
*comparison* — deciding whether the files that appeared are the right ones. The live checks assert on tool messages and token counts, never on
model prose. Nothing grades an answer.

The reason is scope: verification is "did the agent do the thing", and the thing is TBD. A rule,
a rendered artifact or a judge all need a task to be about, and inventing one to have something to
check would be the same YAGNI this file spends a section refusing.

Two consequences worth stating rather than discovering:

- **Hardening the harness has a ceiling until the domain lands.** Bounds, capabilities, contracts
  and observability are all reachable now. "Did it succeed?" is not.
- **When the domain does land, this is the first thing to build**, and it belongs in code, not in
  the prompt — preferably a deterministic rule (an exit code, a schema, a read-back of the object
  the agent claims to have created). `deepagents.RubricMiddleware` exists and is the *weakest*
  option: an LLM grading the agent it is part of, which inherits the same wrong assumptions.

One thing already points the right way: **F27 turned "assert on tool messages, not prose" from good
practice into the vendor's own instruction**, because gpt-oss's chain of thought is unsupervised and
may contain what the answer was told to omit. Whatever verification arrives later reads the tool
record, not the narration.

**The gate.** `scripts/check.sh` is the only copy of the sequence; the pre-commit hook and
`.github/workflows/ci.yml` both call it and re-list nothing. **The hook is installed here**, but not
automatically — re-run `uv run pre-commit install` in a fresh clone, never from a worktree. `main`
is protected and **a direct `git push` is rejected**, behind three required checks named exactly
`gate`, `Analyze (python)` and `Analyze (actions)`; rename the `gate` job and every PR becomes
unmergeable in the same instant. Full reasoning: `docs/findings.md`, "Repo gates".

## Verified API facts

Recorded from `inspect` against the installed wheels, 2026-09-17/18. **The single copy** —
`docs/findings.md` points here. Re-verify both after any `uv sync` that moves a version.

```
deepagents 0.7.15  langchain 1.4.1  langchain-core 1.6.3  langgraph 1.2.11  openai 3.14.1
langchain-openai 1.6.2  langsmith 0.12.6  weave 0.53.9  Python 3.13
pytest 9.1.1  mypy 2.3.1  ruff 0.16.8  pyright 1.1.414
```

**`langchain_openai.ChatOpenAI`, not `init_chat_model`.** The latter infers a provider from the
model string, and `org/model:provider` router ids are meaningless to it. `ChatOpenAI`'s constructor
takes *aliases*: `model` → `model_name`, `base_url` → `openai_api_base`, `api_key` →
`openai_api_key` (coerced to `SecretStr`), `timeout` → `request_timeout`; `max_retries` and
`temperature` have none.

**`deepagents.create_deep_agent`** takes positional `model`, `tools` and 16 keyword-only parameters,
and returns a `CompiledStateGraph`. That list, `BackendProtocol` and `SubAgent` are in
`docs/findings.md`, "deepagents API surface". Three constraints bite in practice:

- `skills=[...]` needs a real backend (e.g. `FilesystemBackend`); it silently loads nothing
  otherwise, and skills are **not** inherited by subagents — pass `skills` on each subagent spec.
- A consistent `thread_id` shares a conversation only *once a `checkpointer` exists*. There is none,
  so langgraph retains nothing between `invoke` calls and a conversation continues by sending prior
  messages back: `run_turn(agent, prompt, history=...)` returns exactly what the next call wants — *of the
  messages*. **The filesystem does not carry.** `run_turn` sends only `messages`, so the agent's
  `StateBackend` is empty at the start of every turn: a file written in turn one is gone in turn
  two. That makes `TurnResult.files` precisely "what this turn wrote", and makes multi-turn work on
  the agent's own files impossible until a `checkpointer` is set (F43).
- The general-purpose subagent behind `task` gets its own middleware, not the parent's (F20).

### Hugging Face router via `langchain-openai`

Base URL `https://router.huggingface.co/v1`, auth via `HF_TOKEN`. Model ids are `org/model`,
suffixed to steer routing: `:provider`, or the policies `:fastest`, `:cheapest`, `:preferred`.

**The default is pinned: `openai/gpt-oss-120b:groq`. Omitting the suffix is not "no routing
decision" — the router reads it as `:fastest`.** That ranks the eleven live providers by first-token
latency, so every caller on the default lands on the same one or two, which is where the queues
fill. Measured 2026-09-23: **3 of 127 calls returned `429 queue_exceeded` unpinned, 0 of 130
pinned**, and the router sets `x-should-retry: false` so the openai client will not retry them —
`max_retries` is irrelevant to that failure (F46). `main.ROUTING_POLICY_SUFFIXES` is the list the
F25 check treats as "not a pinned provider"; a named provider narrows that check to that provider,
and an unserved pin is refused rather than narrowed to an empty set.

Pinning was already wanted for two other reasons, and one literal buys all three. gpt-oss-120b
natively supports 128k, but the live providers do not agree: most advertise 131072, baseten
advertises 128072, and two advertise nothing at all (F25) — so **the context window is a number
rather than a range** only when pinned. And unpinned routing means the weights answering are
whatever that provider is serving, which for open weights OpenAI's own worst-case evaluation showed
can be fine-tuned into a non-refusing model: **a trust decision as much as a reproducibility one.**

`groq` on the catalogue read that day: second-highest throughput of the eleven, a stated 131,072
window comfortably above `CONTEXT_WINDOW_TOKENS`, tools *and* structured output, and deliberately
**not** the latency leader — `cerebras` is, and was identified as one of the two backends serving
the unpinned runs. Alternate model `deepseek-ai/DeepSeek-V4-Flash` (3 providers). Note that
`api.endpoints.huggingface.cloud` is the *control plane*, not an inference base URL (F7).

**Pin `use_responses_api=False` explicitly; do not leave it unset.** The default `None` does *not*
mean Chat Completions — langchain infers the endpoint from the model name and payload, **independent
of `base_url`**, and the router serves `/v1/chat/completions` only. `model.py` pins it via
`USE_RESPONSES_API` and asserts it as a postcondition (F1).

**Reasoning is on by default and costs real tokens** — often most of the output budget, so a token
cap is mostly a reasoning cap. The dial is `reasoning_effort`, a `ModelConfig` field defaulting to
`None` and settable per run with `REASONING_EFFORT`. **Three levels, not six:** gpt-oss was
post-trained on `low`, `medium` and `high` and carries the level in the system message under
harmony; the router documents `none`, `minimal` and `xhigh` too and answers 400 for all three, one
of them from the model's own chat template. Unset is not "off" and not "between" — it measured
token-for-token identical to `medium` (F26). **Do not use `ChatOpenAI.reasoning`** — that dict field is the Responses API's
and would reroute the request (F17). Body names otherwise match the standard OpenAI ones, except
that `ChatOpenAI` renames `max_tokens` to `max_completion_tokens` on the wire; the router accepts
the renamed key, and `ModelConfig` has no token-cap field (F2).

### Observability

**LangSmith is ambient**: set `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` and the
whole stack traces with no code change — but the value must be the exact literal string `"true"`,
the lookup is `lru_cache`d so a read before `load_dotenv()` sticks for the process, and the legacy
`LANGCHAIN_*` names are **not** inert (F13). **Weave** installs a `WeaveTracer` callback globally
via its LangChain autopatch integration, gated on `WEAVE_TRACE_LANGCHAIN`. Both run at once (F11),
and `available_backends()` activates every configured backend rather than selecting one.

## Repo layout

Everything in this table lives in `src/my_agent/`.

| Module | What it holds |
|---|---|
| `model.py` | `ModelConfig`, `build_model`, router defaults, `USE_RESPONSES_API`. Reads `os.environ` via `from_env`; the only module that knows the router exists. |
| `agent.py` | `AgentConfig`, `build_agent`. Takes a `BaseChatModel` and imports nothing from `model.py` — `main.py` is the only place the two meet. Compiles **twice**: the second build is what puts a step limit on the `task` subagent (F24). |
| `capabilities.py` | The allowlist and the proof it held: `DEFAULT_FILESYSTEM_TOOLS`, `least_privilege_filesystem`, `compiled_tools`, `compiled_tool_names`, `subagent_graphs`, `bound_step_limit`, `require_withheld`/`require_granted`, `compiled_output_keys`, plus the bounds on granted capabilities (`PARENT_STEP_LIMIT`, `SUBAGENT_STEP_LIMIT`, `GREP_MATCH_LIMIT`, the eviction limits, `bounded_compaction` and `CONTEXT_WINDOW_TOKENS`) and the `DEEPAGENTS_PLUGIN_GROUPS` / `DEEPAGENTS_CACHING_PROBE_MODULES` / `HARNESS_PROFILE_FIELDS` pins plus `require_no_harness_profile` (F47). |
| `contracts.py` | `check_config_contract`, `pydantic_param_names` — the import-time check that makes `as_kwargs()` splatting safe. |
| `run.py` | `Invokable`, `RunBounds`, `RunDeadline`, `RunTokenBudget`, `TurnResult`, `run_turn`, `resume_turn` — one bounded turn, with three outcomes: finished, failed (`DeadlineExceeded`, `StepLimitExceeded`, `TokenLimitExceeded`, `ResumeLimitExceeded`, all subclasses of `BoundExceeded`) or paused for approval, `failed_tool_calls` for a turn that finished without doing what it says, `answered` for a turn cut off before its answer began, and `state` — every key the graph returned bar `messages`/`__interrupt__`, with `files` and `structured_response` as properties over it. `_invoke` used to read `messages` and drop the rest (F40, F43). Owns the step limit, the wall clock and token budget across a pause, the resume count, the thread and multi-turn history. Imports no deepagents and builds no model. |
| `main.py` | `uv run my-agent` — the composition root, and the eight live checks (`CheckOutcome`, `live_check_repeats`, pass^k). |
| `negative_space.py` | `require`/`unreachable`/`bounded`, and the only doctests in `src/`. |
| `tracing.py` | `TracingBackend`, `LangSmithTracing`, `WeaveTracing`, `available_backends`, `langchain_tracer_names`. |
| `mirror.py` | `JsonlMirror`, `run_log_path`, `mirror_to_file` — the always-on local JSONL mirror of every agent event, including the per-call request size, a per-run per-tool byte breakdown (F30) and the server's retry advice on a failed model call (F46). |
| `usage.py` | `call_usage` — the one reading of a model call's `usage_metadata`, shared by `RunTokenBudget` and `JsonlMirror` so the bound and the log cannot disagree. |

`tests/` mirrors that one file per module, offline by default, plus `conftest.py` for shared
fixtures and the socket guard. The only `-m live` tests are one each at the end of `test_tracing.py`
(calls the real `weave.init()` and hits the router) and `test_run.py` (proves multi-turn history
against a real graph, which a fake cannot show). `evals/` is empty. `docs/findings.md` holds F1–F47
plus the repo-gates, deepagents-surface, test-infrastructure and observability appendices;
`scripts/audit_negative_space.py` is **vendored** from the negative-space-programming skill — do not
hand-edit it, refresh by re-copying (it is excluded from ruff and mypy).
