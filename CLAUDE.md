# CLAUDE.md — Bayesian IF

## What This Is

A Bayesian decision-theoretic Interactive Fiction agent that uses the `credence` library
for information-gathering decisions. The agent uses VOI (Value of Information) to decide
which sources to consult (look, examine, inventory, LLM) before committing to an action.

## Architecture

- `BayesianAgent` from credence is the **information-gathering controller** — it decides
  which sources to consult, not which action to take directly.
- Each game step is a fresh "question" — answer posteriors reset per step, but the
  reliability table persists across steps, learning which sources work in which situations.
- Score deltas provide ground truth: `delta > 0` → correct, `delta < 0` → wrong, `delta == 0` → no update.
- Info tools use save/restore to peek at game state without consuming a turn.

## Design Principles

Same as credence: everything is EU maximisation, no hacks, LLM outputs are data.

## Dependencies

- `credence` — Bayesian inference layer (path dependency during development)
- `jericho` — Z-machine IF interpreter (optional, for real games)
- `textworld` — Procedural IF environments (optional)
- `httpx` — Ollama HTTP client (optional, for LLM advisor tool)
