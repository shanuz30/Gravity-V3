# Runtime Pipeline: `antigravity/runtime.py::run_end_to_end`

*What Stage 1 actually proved, and what it explicitly did not.*

This document describes `antigravity/runtime.py::run_end_to_end()`, the function that
wires the Antigravity (Track B) reasoning core to the Gravity-V3 (Track A) execution
shell into one callable chain, and `tests/test_runtime.py`, the test suite that exercises
it. See `docs/coherence_report.md` for the broader two-track history this closes a gap
in.

The chain proven to execute for real is:

```
Antigravity.run()
    -> bridge.try_to_reasoning_assessment()
    -> execute_session_loop_with_fallback()
           (internally: decide_final_action(), then the writer/auditor loop,
            then the mechanical gate)
    -> a structured runtime result dict
```

Only `anthropic`'s `messages.create` is ever mocked by the tests that exercise this
chain. Everything else described below as "proven" ran its real, unmocked code path
during a real test execution — nothing here is inferred from reading the source alone.

---

## 1. Proven behavior

Each claim below is backed by a specific test in `tests/test_runtime.py`.

- **`Antigravity.run()` executes for real and produces real signals**, not canned
  values — `test_real_antigravity_run_produces_real_signals` asserts an exact
  `confidence` value (0.92) and `alarm` ("SAFE") that only come out of real
  `GANSignal`/`CRAGSignal` computation, and separately confirms the mocked
  `anthropic` client was actually called at least 3 times (Generator, Discriminator,
  Convergence). Note precisely what this test does and doesn't check: `alarm=="SAFE"`
  requires the real `CRAGDetector` to have found `js_divergence<=0.05` and
  `cosine_sim>=0.70`, but the test doesn't assert those two raw numbers directly —
  `ReasoningAssessment` doesn't expose them — only that their real computation landed
  in the `SAFE` tier.
- **The bridge conversion runs for real** — every test that inspects
  `result["assessment"]` (e.g. `test_safe_recipe_yields_answer_success`,
  `test_full_drift_recipe_yields_blocked`) is observing a real
  `ReasoningAssessment` built by `bridge.try_to_reasoning_assessment` from the real
  `PipelineState` `Antigravity.run()` returned, not a stub.
- **`decide_final_action` (Gravity-V3's policy authority) runs for real and its
  decision reaches the caller** — `test_safe_recipe_yields_answer_success` (`ANSWER` ->
  `"answer"`/`"ANSWER_SUCCESS"`), `test_full_drift_recipe_yields_blocked` (`FULL_DRIFT`
  alarm -> `"block"`/`"BLOCKED"`), `test_low_confidence_recipe_yields_antigravity_escalation`
  (low confidence -> `"escalate"`/`"ANTIGRAVITY_ESCALATION"`), and
  `test_retrieve_tier_reachable_with_empty_evidence_ids` (no `evidence_ids` ->
  `"retrieve"`/`"RETRIEVAL_REQUIRED"`) each drive a different branch of the real
  decision rule in `scripts/coordinator_agent.py::decide_final_action` /
  `scripts/reasoning_assessment.py::ReasoningAssessment.create`.
- **`execute_session_loop_with_fallback` runs for real, including the writer** —
  `test_safe_path_reaches_the_writer` confirms the retrieval text appears in the
  returned draft, and `test_writer_generate_actually_invoked_via_spy` proves this by
  spying on a caller-supplied writer object and asserting its real `generate()` was
  called exactly once with the arguments session_loop passed.
- **The real `TruthSentryAuditor` runs and its output is surfaced** —
  `test_audit_field_contains_real_truth_sentry_output` and
  `test_default_recording_auditor_wraps_real_truth_sentry` both assert
  `result["audit"]["final_audit_result"] == "[TRUTH_VERIFIED]"`, a string only the real
  auditor produces (session_loop's own return value never exposes this on its own,
  which is why `RecordingAuditor` exists — see Section 3).
- **The mechanical gate runs for real and can independently reject a draft** —
  `test_mechanical_gate_rejects_blacklisted_draft` drives a draft containing
  `"DROP TABLE users"` through the real `MechanicalGate`, and asserts the result is
  `status="BLOCKED"`/`reason="mechanical_gate_rejected"` while `policy_action` is still
  `"answer"` — proving this is a distinct rejection layer from the coordinator's policy
  decision, not the same block reported twice.
- **`chat_history_store`, when supplied, receives a real entry with the correct final
  status** — `test_chat_history_receives_entry_with_final_status` and
  `test_chat_history_records_blocked_status` write to a real `ChatHistoryStore` backed
  by a temp file and read the entry back.
- **No hidden network calls escape the mocked `anthropic` boundary** —
  `test_no_network_call_outside_mocked_anthropic_boundary` asserts the lazy
  `sentence_transformers` import is never triggered, because `run_end_to_end` always
  supplies its own deterministic `embed_fn` (see Section 2).
- **The global-injection save/restore is verified, not just assumed** —
  `test_global_injection_is_restored_after_success`,
  `..._after_short_circuit`, and `..._after_mechanical_gate_rejection` each snapshot
  every module-level collaborator on `scripts.session_loop` before and after a call and
  assert they're byte-identical.
- **The RETRIEVE tier is reachable and short-circuits before the writer** —
  `test_retrieve_tier_reachable_with_empty_evidence_ids` passes `evidence_ids=()` and
  confirms `audit` is `None` (writer/auditor never ran).
- **`trace_id` defaults to a generated UUID, or echoes a supplied `session_id`** —
  `test_trace_id_present_and_defaults_to_generated_uuid` and
  `test_trace_id_uses_session_id_when_provided`.
- **The Tier-3/malformed-assessment short-circuit blocks correctly and never invokes
  session_loop** — see Section 5, which covers this in detail as both a proven
  behavior and a documented limitation.

## 2. Deterministic test assumptions

These are stand-ins the test harness supplies so the real pipeline can run
deterministically in a test environment, without a real knowledge graph, writer
pipeline, or embedding model. They are documented in full in `runtime.py`'s module
docstring (notes 1–2); this section restates them for a reader who hasn't seen that
file.

- **Knowledge-graph stand-in.** There is no real retrieval or knowledge-graph
  implementation anywhere in this repo. `retrieval_context` — the second positional
  argument to `run_end_to_end` — is a **plain string**. The `query_knowledge_graph`
  collaborator installed on `scripts.session_loop` for the duration of the call
  (`_make_query_knowledge_graph` in `runtime.py`) is a trivial closure that always
  returns `{"canonical_facts": [retrieval_context], "constraints": []}`, regardless of
  what `user_input` session_loop calls it with.
- **Deterministic writer.** No real writer pipeline exists in this repo either.
  `_DeterministicWriter.generate()` ignores the `prompt`/`facts` arguments session_loop
  passes it and always returns a draft baked in at construction time: the text of the
  *real* Antigravity GAN output (`gan.residual_truth`, falling back to
  `gan.core_claim`) captured from the real `Antigravity.run()` call that already
  happened earlier in the same `run_end_to_end` call. `regenerate()` is a trivial echo;
  none of the tests drive session_loop into its conflict-retry path, so it exists only
  so that code path is well-formed if it's ever reached.
- **Deterministic embedding function, and why the obvious fallback couldn't be used.**
  `run_end_to_end` always supplies its own `_deterministic_embed` to `Antigravity(...)`
  — a bag-of-words counting embedding (hash each word into one of 384 dimensions,
  count, normalize) rather than the hash-seeded-noise fallback that
  `antigravity/core/crag.py::CRAGDetector.embedding_from_text` uses when
  `sentence-transformers` isn't installed. That noise fallback was tried and rejected:
  it empirically produces near-random cosine similarity between *any* two different
  strings, which makes it impossible to reliably land a test recipe in a specific C-RAG
  alarm tier (`SAFE` vs. `FULL_DRIFT`) on purpose. The bag-of-words embedding instead
  gives semantically related or identical strings genuinely higher cosine similarity
  than unrelated ones, so the mocked-`anthropic`-response "recipes" in
  `tests/test_runtime.py` can deterministically target a specific alarm tier by
  choosing whether the GAN's claim text overlaps with the retrieval text.

## 3. The mocked boundary

**Only `anthropic`'s `client.messages.create` is ever mocked**, via a `MagicMock` with a
`side_effect` that inspects the `system` prompt to decide whether to return a
Generator, Discriminator, or Convergence-shaped response (`make_mock_client` in
`tests/test_runtime.py`).

Every other collaborator on the chain runs its real, unmocked code:

| Collaborator | Module |
|---|---|
| `Antigravity` | `antigravity/orchestrator.py` |
| `GANLoop` | `antigravity/core/gan.py` |
| `CRAGDetector` | `antigravity/core/crag.py` |
| `LCVModule` | `antigravity/core/lcv.py` |
| `bridge` (`try_to_reasoning_assessment`) | `antigravity/bridge.py` |
| `coordinator_agent` (`decide_final_action`, `prepare_worker_brief`) | `scripts/coordinator_agent.py` |
| `session_loop` (`execute_session_loop_with_fallback`) | `scripts/session_loop.py` |
| `MechanicalGate` | `scripts/mechanical_gate.py` |
| `TruthSentryAuditor` | `scripts/truth_sentry.py` (wrapped by `runtime.py`'s `RecordingAuditor` only to expose its raw result string, which session_loop's own return value doesn't surface — the underlying check logic is untouched) |
| `ChatHistoryStore` / `record_session_result` | `scripts/chat_history.py` (in the tests that pass `chat_history_store=`) |

`runtime.py`'s own `_DeterministicWriter` stands in for the writer pipeline (Section 2)
and is not itself "mocked" in the unittest-mock sense — it's a real, if minimal, object
whose `generate()`/`regenerate()` methods are actually called by the real
`session_loop` code.

## 4. Real runtime dependencies

Per the root `requirements.txt`:

```
numpy>=1.24
scipy>=1.10
anthropic>=0.34
```

These back `antigravity/core/crag.py` (`numpy`, `scipy.spatial.distance.jensenshannon`),
`antigravity/core/lcv.py` (`numpy`), and `antigravity/orchestrator.py` (`numpy`,
`anthropic`). `scripts/` (the Gravity-V3 side: `session_loop.py`,
`coordinator_agent.py`, `mechanical_gate.py`, `truth_sentry.py`,
`reasoning_assessment.py`, `chat_history.py`) is deliberately pure standard library and
needs nothing from this file — see `docs/coherence_report.md` §1 for why that boundary
is kept. `sentence-transformers` is listed as an optional dependency for a real
embedding model, but `run_end_to_end` never touches it (Section 1,
`test_no_network_call_outside_mocked_anthropic_boundary`).

## 5. Not-yet-implemented retrieval orchestration

**This milestone does not prove anything about Qdrant, Memgraph, or any production
retrieval system.** No such system is invoked, mocked, or referenced anywhere in
`runtime.py` or `tests/test_runtime.py`. `retrieval_context` is, and is only, a plain
Python string supplied by the caller — see Section 2's knowledge-graph stand-in. The
`mcp-servers/lxDIG-MCP` Docker infrastructure documented elsewhere in this repo
(`docs/coherence_report.md` §1, Track A) is unrelated to and untouched by this chain.
Nothing in this document should be read as implying retrieval orchestration is wired up
— it is not.

## 6. Known limitations

Stated plainly, not buried in a footnote.

### (a) Global-injection concurrency risk

`execute_session_loop_with_fallback` reads its collaborators
(`query_knowledge_graph`, `writer_agent`, `auditor_agent`, `coordinator_agent`,
`chat_history_store`, `mechanical_gate`) from module-level globals on
`scripts.session_loop` — that's the only extension point session_loop offers.
`run_end_to_end` saves those globals, overwrites them for the duration of one call, and
restores the saved values in a `finally` block (the same pattern session_loop's own
test suite uses).

**This means `run_end_to_end` is not safe to call concurrently from multiple threads in
the same process.** Two overlapping calls racing inside the same interpreter can clobber
each other's collaborators mid-flight — one call's writer could run against another
call's knowledge-graph closure. `tests/test_runtime.py` verifies the save/restore
round-trips correctly for a single call
(`test_global_injection_is_restored_after_success`, `..._after_short_circuit`,
`..._after_mechanical_gate_rejection`, `test_global_injection_untouched_on_tier3_short_circuit`)
but does **not** test concurrent calls — there is no test proving safety under
concurrency, because there isn't any. Callers running this from multiple threads against
the same process must serialize calls externally (e.g. a lock around the whole
`run_end_to_end` call).

### (b) Tier-3 / malformed-assessment collapse

Inside `Antigravity.run()`, a Tier-3 interrupt (`gan_signal.tier3_risk` is true) breaks
the correction loop before `state.crag` or `state.lcv` are ever assigned
(`antigravity/orchestrator.py`, lines ~150–157). `bridge.to_reasoning_assessment` raises
`AntigravityAssessmentError` whenever `state.gan`, `state.crag`, or `state.lcv` is
`None`, and `try_to_reasoning_assessment` catches *any* exception and returns `None`
instead of raising.

The consequence: **a Tier-3 interrupt and every other cause of a missing or invalid
signal are indistinguishable at the `ReasoningAssessment` boundary** — both produce
`assessment=None`. `run_end_to_end` does not (and by design cannot, without inspecting
`state.alarm` itself, which it deliberately does not do — see below) tell you which one
happened.

`run_end_to_end` treats `assessment=None` as a hard short-circuit to `BLOCK`, calling
`decide_final_action(None)` itself and returning before `execute_session_loop_with_fallback`
is invoked at all. `tests/test_runtime.py`'s `tier3_recipe()` drives this path (a mocked
Convergence response with `"tier3_risk": true`), and four tests confirm it end to end:
`test_tier3_collapse_produces_none_assessment`, `test_tier3_collapse_blocks_not_answers`,
`test_tier3_collapse_never_invokes_the_writer` (proves session_loop itself was never
called, via a spy writer that recorded zero `generate()` calls), and
`test_tier3_collapse_recorded_to_chat_history`.

**This block-on-`None` behavior was not the first implementation, and its absence was a
real bug caught during merge review.** `scripts/session_loop.py`'s own guard is
`if reasoning_assessment is not None:` — it never calls `decide_final_action` at all for
a `None` input. The first version of this wiring passed `assessment` straight through to
`execute_session_loop_with_fallback` unconditionally; for `assessment=None` that guard
is simply skipped, and session_loop falls through into the ordinary retrieval/write/audit
path, exactly as if no assessment had been requested at all. That treats "Antigravity's
output was unusable" identically to "no assessment was ever requested" — the opposite of
Gravity-V3's founding rule that a malformed assessment should BLOCK. The fix (commit
`33286d5`, "Fix: assessment=None must BLOCK, not fall through to answer") makes
`run_end_to_end` itself call `decide_final_action(None)` for exactly this one case —
the only input session_loop's own guard would never call that function for. This is why
the review stage matters: the code executed without error either way, and only a test
asserting the *status* of the `None`-assessment path, not just that it ran, would catch
the difference.

## 7. How to run it

Calling the function directly (only `anthropic_client` needs mocking or a real client):

```python
from antigravity.runtime import run_end_to_end

result = run_end_to_end(
    hypothesis="some hypothesis",
    retrieval_context="the antigravity system converges to zero drift ...",
    anthropic_client=my_anthropic_client,  # or a MagicMock in tests
)
# result: {"status", "assessment", "policy_action", "session_result", "audit", "trace_id"}
```

Running the proof suite and the LCV convergence check:

```bash
python -m unittest discover -s tests -v
python validate.py
```
