# Room Activity Coherence Report

*Synthesis of everything built in this repo, and a diagnosis of how the pieces relate.*

## 1. What's actually in this repo

Reading the full commit history and every file shows **two separate architectural
efforts sharing one repo**, with no code, imports, or shared vocabulary connecting them.

### Track A — "Gravity-V3" (the earlier, shipped-looking track)

Commits `e9bc387` → `7e5ebdc` → `55dc8b4` → `c025707` → `ba1a3f1`, plus the fallback-mechanism
and worker-brief PRs merged later.

| File | Role |
|---|---|
| `scripts/mechanical_gate.py` | Regex blacklist (`rm -rf /`, `DROP TABLE`, …) that intercepts an LLM's execution intent before it reaches a shell/DB — deterministic veto, not a model. |
| `scripts/query_arbitrator.py` | Keyword-based router between "System 1" (semantic/Qdrant) and "System 2" (structural/Memgraph) queries. |
| `scripts/metacognitive_monitor.py` | `EmotionVectorController` — hand-coded `if/elif` on event type (`MODEL_SWITCH_HIGH_PRO`, `API_TIMEOUT_OR_FAILURE`, …) that nudges a hallucination-risk threshold and grounding strictness. |
| `scripts/coordinator_agent.py` | Builds a `WORKER_BRIEF` (canonical facts + constraints + task scope) from a knowledge-graph payload; refuses with `INSUFFICIENT VARIABLES` if required context is missing. |
| `scripts/truth_sentry.py` | Falsification-first auditor that checks a draft against the same knowledge-graph payload / brief and returns `[LOGIC_ERROR]` or `[CONFLICT_FOUND]`. |
| `scripts/session_loop.py` | Wires coordinator → writer → truth-sentry into a write/audit loop with escalation instead of infinite retry. |
| `scripts/chat_history.py` | JSON-backed `ChatHistoryStore` with keyword search, called from `session_loop`. |
| `mcp-servers/lxDIG-MCP/docker-compose.yml` | Memgraph + Qdrant + a Node MCP bridge — the actual infra the arbitrator and gate assume. |
| `docs/architecture.md`, `docs/setup_guide.md`, `docs/krones_proof_of_sovereignty.md` | External-facing pitch naming **Gemini 3.1 Pro** as the core model, WSL2/Docker as the "sovereign" runtime, and Krones.digital as the target audience. |

This track has unit tests (`tests/test_*.py`) for `coordinator_agent.py`, `truth_sentry.py`,
`session_loop.py`, and `chat_history.py` — `mechanical_gate.py`, `metacognitive_monitor.py`,
and `query_arbitrator.py` have none. Every script is pure standard library, though — no
external dependency needed to run it.

### Track B — "Antigravity" (the later, more rigorous track)

Single commit `d397fbd` — "Add Antigravity architecture: LCV, C-RAG, GAN loop, orchestrator" —
plus this repo's own `CLAUDE.md`, which documents *only* this track.

| File | Role |
|---|---|
| `antigravity/core/types.py` | Shared dataclasses (`AnchorState`, `GANSignal`, `CRAGSignal`, `LCVOutput`) that every other module passes through. |
| `antigravity/core/gan.py` | Generator → Discriminator → Convergence adversarial loop against a real LLM (`anthropic` client), producing a confidence score. |
| `antigravity/core/crag.py` | Two-layer drift detector: Jensen-Shannon divergence on token distribution (Layer 1) + cosine similarity to a retrieval anchor (Layer 2). Both must fire for a `FULL_DRIFT` alarm. |
| `antigravity/core/lcv.py` | The Learned Correction Vector, with the **gamma-damping fix**: `v* = min(eps, gamma·d) · normalize(r−x)`. This is the mathematically load-bearing piece — see below. |
| `antigravity/orchestrator.py` | Ties retrieval → GAN → C-RAG → LCV into one loop, with "Clock-Time" batching to prevent noise-driven oscillation and a Tier-3 interrupt for recursive spirals. |
| `validate.py` | The one executable proof: without `gamma·d` damping the correction has an unstable fixed point at `d*≈0.17` (matching the GAN's empirical ~71% confidence ceiling); with `gamma=0.85` it's globally convergent in ~60 iterations. |

`CLAUDE.md`'s own status list says persistence (`memory.md`/`wiki.md`), real logprobs for
C-RAG Layer 1, a MySQL swarm blackboard, and Wolfram verification are **not yet built** —
confirmed here: `antigravity/state/` contains only a `.gitkeep`, and
`orchestrator.py::_pseudo_token_dist` is explicitly a hash-based placeholder, not real logprobs.

### Environment check (read-only, this sandbox only)

`numpy`, `scipy`, `anthropic`, and `pytest` are not installed here, so:
- `python3 validate.py` fails on `import numpy` before it can run any of its convergence proof.
- Track B's code (all of `antigravity/`) cannot execute in this environment at all right now.
- Track A's scripts have no external imports, so they should run, but their `tests/` can't be
  collected without `pytest`.

This isn't a code defect — the repo has no `requirements.txt` yet for either track — but it
means neither track's "tested" claims can currently be re-verified in-session.

## 2. Diagnosis: are these one project or two?

They're **the same underlying problem, solved twice, at very different levels of rigor** —
not a random duplication, but not one coherent system either yet.

Both tracks exist to answer the same question: *how does an LLM-driven agent avoid confidently
saying or doing the wrong thing, and how does it know when to correct itself versus escalate?*

- Track A's answer is a set of **hand-tuned heuristics**: a keyword list routes queries, an
  `if/elif` chain nudges risk thresholds by fixed deltas per event type, and a regex blacklist
  vetoes destructive commands. It's shippable now and needs no proof — but the thresholds are
  arbitrary and there's no argument for why they're stable under repeated correction.
- Track B's answer is the **same shape of mechanism, formalized**: C-RAG's `alarm` property is
  a principled version of "should I trust this and correct, or not" that Track A's Emotion
  Vectors approximate by hand; LCV's gamma-damping is a proven fix for exactly the oscillation
  failure mode that an ungoverned correction loop like Track A's would be vulnerable to (Track A
  has no equivalent stability argument for its threshold nudges).

There's no evidence in the code or commit history that Track B was written *as* a replacement
for Track A — `d397fbd` doesn't touch or reference any Track A file, and `CLAUDE.md` was
written as if Track A doesn't exist. This currently reads as parallel, disconnected work rather
than a deliberate sequel.

### Recommendation

Treat **Antigravity (Track B) as the reasoning core, and Gravity-V3's scripts (Track A) as the
shippable shell around it** — don't merge the file trees, but plan the following concrete
substitutions when Track B matures past its "not yet built" gaps:

1. `metacognitive_monitor.py`'s `EmotionVectorController._apply_vector` hand-coded deltas should
   become a thin adapter that reads `CRAGSignal.alarm` and `LCVOutput.damping_active` instead of
   branching on event-type strings — Track B already computes the signal Track A is
   approximating.
2. `query_arbitrator.py`'s System-1/System-2 keyword routing is a reasonable, cheap gate to keep
   as-is *upstream* of the Antigravity pipeline (deciding what to retrieve), rather than
   something Track B needs to reimplement.
3. `mechanical_gate.py` and `truth_sentry.py`/`coordinator_agent.py`'s WORKER_BRIEF pattern stay
   independent — they're about command safety and fact-grounding, which is orthogonal to drift
   correction and doesn't overlap with either C-RAG or LCV.
4. The Krones pitch docs (`docs/krones_proof_of_sovereignty.md`, `docs/architecture.md`) should
   be treated as a point-in-time external artifact, not living architecture docs — they
   describe Gemini 3.1 Pro + Memgraph/Qdrant, which is Track A's world, and will go stale
   the moment Track B's substitutions above land.

### Status update

Recommendation #1 is implemented. Rather than importing Track B's dataclasses directly into
`metacognitive_monitor.py` (which would give a zero-dependency script a hard dependency on
numpy/scipy/anthropic), the two tracks now meet at a small, immutable, stdlib-only boundary
type: `scripts/reasoning_assessment.py::ReasoningAssessment` (confidence, risk_score, alarm,
damping_active, a bounded `recommended_action` enum, evidence tracking). `antigravity/bridge.py`
is the one file that translates a Track B `PipelineState` into that contract; the dependency
direction is one-way (Antigravity depends on `scripts/`, never the reverse). Gravity-V3's own
`coordinator_agent.py::decide_final_action` remains the actual policy authority over what an
assessment leads to, and `session_loop.py` now calls it before writing anything, short-circuiting
on `BLOCK`/`ESCALATE`/`RETRIEVE`/`CLARIFY` — additively, with `reasoning_assessment=None` (the
default) preserving the exact prior behavior. This also closed a gap section 1 flagged:
`mechanical_gate.py` was never actually invoked from `session_loop.py` before this change; it's
now an optional hook checked on the successful-draft path, independent of the CRAG/LCV signals
themselves (per recommendation #3, it stays a plain command-safety check). All 61 tests pass
(46 pre-existing/added-with-the-adapter + this integration's own), and `validate.py`'s LCV proof
was re-run unchanged before and after.

## 3. Open gaps (carried from `CLAUDE.md` + this session's findings)

- No `memory.md`/`wiki.md` persistence wired yet — `antigravity/state/` is empty.
- C-RAG Layer 1 runs on a hashed pseudo-distribution, not real token logprobs.
- No MySQL blackboard for swarm/multi-agent operation.
- ~~No `requirements.txt`/dependency manifest for either track~~ — a root `requirements.txt`
  now covers Track B's `numpy`/`scipy`/`anthropic` (Track A stays dependency-free by design, so
  it needs none). `pytest` is still absent from this sandbox, but the whole suite runs fine
  under stdlib `unittest` instead — see `python3 -m unittest discover -s tests`.
- `CLAUDE_CODE_PROMPTS.md` already lays out five ready-to-run implementation prompts for closing
  the remaining Track B gaps in order — that file is the actual next-actions list for those.

## 4. Runtime path proof

Stage 1 of this integration added `antigravity/runtime.py::run_end_to_end()`, which
wires the full chain — `Antigravity.run()` (Track B) `-> bridge.try_to_reasoning_assessment()
-> execute_session_loop_with_fallback()` (Track A: `decide_final_action`, the real
writer/auditor loop, the mechanical gate) `->` a structured result — into one callable,
and proved with `tests/test_runtime.py` that it executes for real. Only `anthropic`'s
`messages.create` is mocked; `GANLoop`, `CRAGDetector`, `LCVModule`, the bridge,
`coordinator_agent`, `session_loop`, `MechanicalGate`, and `TruthSentryAuditor` all run
their real, unmocked code. Full detail, including exactly which test proves each claim,
lives in `docs/runtime_pipeline.md` — this section is a pointer and a summary, not a
duplicate.

Two things this milestone deliberately does **not** claim: it does not exercise any
real knowledge-graph or retrieval system (`retrieval_context` is a plain string
stand-in — no Qdrant/Memgraph involvement, consistent with §3's "not yet built" list),
and it does not prove concurrent-call safety — `run_end_to_end` temporarily mutates
`scripts.session_loop`'s module-level globals and restores them in `finally`, which
makes concurrent calls from multiple threads in the same process unsafe. The other
limitation worth carrying forward here: a Tier-3 interrupt inside `Antigravity.run()`
and any other malformed-signal cause collapse to the same `assessment=None` at the
`ReasoningAssessment` boundary, indistinguishable from each other; `run_end_to_end`
now correctly short-circuits that case to BLOCK, a fix made during merge review after
the first implementation let `None` fall through to the ordinary answer path instead
(see `docs/runtime_pipeline.md` §6 for the full account).

This closes the "can these two tracks actually run together, not just share a type
boundary" question left open at the end of §2's status update: yes, for the chain and
under the deterministic test assumptions documented in `docs/runtime_pipeline.md` §2.
