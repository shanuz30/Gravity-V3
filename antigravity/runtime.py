"""
antigravity/runtime.py — Stage-1 runtime wiring.

Proves this chain executes for real, end to end:

    Antigravity.run()
        -> bridge.to_reasoning_assessment()  (via the non-raising
           try_to_reasoning_assessment)
        -> execute_session_loop_with_fallback()
               (internally: decide_final_action(), then the writer/auditor
                loop, then the mechanical gate)
        -> a structured runtime result (this module's own return contract)

Only `anthropic`'s client (`messages.create`) is ever mocked by callers of
this module. Every other collaborator on the chain (Antigravity, GANLoop,
CRAGDetector, LCVModule, bridge, coordinator_agent, session_loop,
MechanicalGate, TruthSentryAuditor) runs its real code path.

--------------------------------------------------------------------------
ASSUMPTIONS / DESIGN NOTES (documented here, and in the task summary)
--------------------------------------------------------------------------

1. Knowledge-graph stand-in. There is no real retrieval / knowledge-graph
   implementation anywhere in this repo. `retrieval_context` is a plain
   string stand-in for it. The `query_knowledge_graph` collaborator
   installed on `scripts.session_loop` for the duration of the call is a
   trivial deterministic closure that always returns
   `{"canonical_facts": [retrieval_context], "constraints": []}`,
   regardless of the `user_input` session_loop calls it with.

2. Deterministic writer. No real writer pipeline exists in this repo
   either. `_DeterministicWriter` (below) is a minimal stand-in. Its
   `generate()` ignores the `prompt`/`facts` args session_loop calls it
   with and returns a draft that was baked in at construction time: the
   text of the *real* Antigravity GAN output (`gan.residual_truth`, falling
   back to `gan.core_claim`) captured from the real `Antigravity.run()`
   call that already happened before session_loop is ever invoked.
   `regenerate()` is a trivial echo — none of this module's own tests drive
   session_loop into its conflict-retry path, so it exists only so that
   code path is well-formed if it is ever reached.

3. `mechanical_gate=None` means "use the real default gate", NOT "skip the
   gate". This is the OPPOSITE of session_loop.py's own module-level
   default (there, `mechanical_gate = None` means "no gate installed, skip
   the check" — pre-existing, frozen behavior). Passing
   `mechanical_gate=None` to `run_end_to_end` constructs and installs a
   real `MechanicalGate()`, because this milestone is specifically about
   proving the mechanical gate really runs. Pass an explicit gate object of
   your own if you want different behavior.

4. `assessment=None` (malformed PipelineState / Tier-3 collapse) short-
   circuits to BLOCK *before* `execute_session_loop_with_fallback` is ever
   called — `session_loop` is not invoked at all in this case. This was
   corrected after review: `session_loop.py`'s own guard is
   `if reasoning_assessment is not None:` — it never calls
   `decide_final_action` for a `None` input, so passing `None` straight
   through does NOT block; it would have silently fallen into the ordinary
   write/audit/mechanical-gate path, treating "Antigravity's output was
   unusable" identically to "no assessment was ever requested." That
   defeats the entire point of this milestone (Antigravity/Gravity-V3's
   founding rule: malformed assessment -> BLOCK).

   The fix: this module calls `decide_final_action(None)` itself — reading
   the frozen `scripts/coordinator_agent.py` contract, never modifying it —
   for exactly the one input session_loop's own guard means it would never
   call that function for itself. This is not "calling it twice on the same
   input" (there is exactly one call site per input: session_loop's, for
   every non-None assessment; this module's, only for the None case they
   are mutually exclusive) and not a re-decision on a value session_loop
   already decided — session_loop never gets the chance to decide on a
   `None` input at all. The short-circuit result records
   `status="BLOCKED"`, `policy_action="block"`, `audit=None`, and a
   `session_result` note explaining session_loop was never invoked; it is
   still persisted to `chat_history_store` when one was supplied, for audit
   parity with every other short-circuit path. This module still never
   inspects `state.alarm` or otherwise special-cases the Tier-3 path itself
   — the assessment=None collapse (Tier-3 and every other malformed-state
   cause look identical here) is accepted exactly as documented in the task
   brief, not worked around.

5. Concurrency / thread-safety. `execute_session_loop_with_fallback` reads
   its collaborators from module-level globals on `scripts.session_loop`.
   This module saves those globals, overwrites them for the duration of one
   call, and restores the saved values in a `finally` block — the same
   pattern session_loop's own test suite uses. This means two concurrent
   (threaded or multiprocessed-but-sharing-this-interpreter) calls to
   `run_end_to_end` racing inside the same process are UNSAFE: they can
   clobber each other's collaborators mid-flight. This is a real limitation
   of the wiring, not a hypothetical one — do not call `run_end_to_end`
   from multiple threads against the same process without external
   serialization (e.g. a lock around the whole call).

6. `evidence_ids`. The milestone brief's headline "required API" signature
   does not list `evidence_ids` as a parameter, but its own implementation
   notes require exercising both the "has evidence" and "no evidence"
   (RETRIEVE) tiers through this function. It is exposed here as an
   additional keyword-only parameter with a non-empty default, so callers
   who don't care about the RETRIEVE tier get ANSWER-eligible behavior by
   default, while tests can pass `evidence_ids=()` to deliberately hit
   RETRIEVE.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

import numpy as np

import scripts.session_loop as session_loop
from scripts.chat_history import record_session_result
from scripts.coordinator_agent import decide_final_action
from scripts.mechanical_gate import MechanicalGate
from scripts.reasoning_assessment import ReasoningAssessment
from scripts.session_loop import execute_session_loop_with_fallback
from scripts.truth_sentry import TruthSentryAuditor

from . import bridge
from .orchestrator import Antigravity

_PLACEHOLDER_API_KEY = "unused-placeholder-not-a-real-key"


def _deterministic_embed(text: str, dims: int = 384) -> np.ndarray:
    """
    Deterministic, dependency-free embedding stand-in.

    Deliberately NOT the hash-seeded-noise pattern used as the ImportError
    fallback in antigravity/core/crag.py::CRAGDetector.embedding_from_text —
    empirically that produces near-random cosine similarity between any two
    different strings, making it impossible to deterministically land in a
    given C-RAG alarm tier. This bag-of-words counting embedding instead
    gives semantically related/identical strings a genuinely higher cosine
    similarity than unrelated ones, so test recipes can reliably target
    SAFE vs. FULL_DRIFT.
    """
    v = np.zeros(dims)
    for w in text.lower().split():
        v[hash(w) % dims] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else np.ones(dims) / np.sqrt(dims)


class _DeterministicWriter:
    """See module docstring, note 2. The only writer pipeline this repo has."""

    def __init__(self, draft: str):
        self._draft = draft
        self.generate_calls: list[dict] = []
        self.regenerate_calls: list[tuple] = []

    def generate(self, *, prompt, facts):
        self.generate_calls.append({"prompt": prompt, "facts": facts})
        return self._draft

    def regenerate(self, draft, audit_result):
        self.regenerate_calls.append((draft, audit_result))
        return draft


class RecordingAuditor:
    """
    Wraps a real TruthSentryAuditor and records its last raw result string,
    so run_end_to_end can surface it in the "audit" field of its return
    value — session_loop.py (frozen) never exposes the auditor's raw string
    on its own success-path return value.
    """

    def __init__(self):
        self._inner = TruthSentryAuditor()
        self.last_result: Optional[str] = None

    def check(self, draft, truth_data, worker_brief=None):
        self.last_result = self._inner.check(draft, truth_data, worker_brief=worker_brief)
        return self.last_result


def _make_query_knowledge_graph(retrieval_context: str) -> Callable[[str], dict]:
    """See module docstring, note 1."""

    def _query(user_input: str) -> dict:
        return {"canonical_facts": [retrieval_context], "constraints": []}

    return _query


def run_end_to_end(
    hypothesis: str,
    retrieval_context: str,
    *,
    anthropic_client=None,
    mechanical_gate=None,
    writer=None,
    auditor=None,
    chat_history_store=None,
    session_id=None,
    evidence_ids: tuple[str, ...] = ("evidence-1",),
) -> dict:
    """
    Run the full chain for real. See the module docstring for the exact
    assumptions this wiring makes about each collaborator.

    Returns a dict with keys: status, assessment, policy_action,
    session_result, audit, trace_id. See the task brief / module docstring
    for the exact contract of each field.
    """
    trace_id = session_id or str(uuid.uuid4())

    # ── 1. Antigravity.run() — real GAN/CRAG/LCV pipeline ──────────────────
    antigravity = Antigravity(
        api_key=_PLACEHOLDER_API_KEY,
        embed_fn=_deterministic_embed,
    )
    if anthropic_client is not None:
        # GANLoop captured its own reference to the default client at
        # Antigravity.__init__ time — both must be replaced or the mock
        # is silently never hit.
        antigravity.client = anthropic_client
        antigravity.gan.client = anthropic_client

    state = antigravity.run(hypothesis=hypothesis, retrieval_context=retrieval_context)

    # ── 2. bridge — non-raising conversion to the Gravity-V3 contract ──────
    assessment: Optional[ReasoningAssessment] = bridge.try_to_reasoning_assessment(
        state,
        evidence_ids=tuple(evidence_ids),
        reasoning_trace_id=None,
    )

    # Malformed/unavailable assessment: short-circuit to BLOCK ourselves,
    # before session_loop is ever invoked. See module docstring, note 4, for
    # why this is the one legitimate case where this wrapper calls
    # decide_final_action() directly rather than letting session_loop do it.
    if assessment is None:
        decision = decide_final_action(None)
        result = {
            "status": "BLOCKED",
            "assessment": None,
            "policy_action": decision.action.value,
            "session_result": {
                "note": (
                    "session_loop was not invoked: Antigravity's output was "
                    "malformed or unusable (missing crag/lcv signals, a "
                    "Tier-3 interrupt, or a bridge conversion failure)."
                )
            },
            "audit": None,
            "trace_id": trace_id,
        }
        if chat_history_store is not None:
            record_session_result(chat_history_store, hypothesis, result, session_id)
        return result

    policy_action = assessment.recommended_action.value

    gan_text = ""
    if state.gan is not None:
        gan_text = state.gan.residual_truth or state.gan.core_claim or ""

    resolved_writer = writer if writer is not None else _DeterministicWriter(gan_text)
    resolved_auditor = auditor if auditor is not None else RecordingAuditor()
    resolved_gate = mechanical_gate if mechanical_gate is not None else MechanicalGate()

    # ── 3. session_loop — real write/audit loop + mechanical gate ──────────
    # session_loop reads its collaborators from module-level globals; there
    # is no other extension point. Save/restore in finally — see module
    # docstring, note 5, for the concurrency caveat this implies.
    saved = {
        "query_knowledge_graph": session_loop.query_knowledge_graph,
        "writer_agent": session_loop.writer_agent,
        "auditor_agent": session_loop.auditor_agent,
        "coordinator_agent": session_loop.coordinator_agent,
        "chat_history_store": session_loop.chat_history_store,
        "mechanical_gate": session_loop.mechanical_gate,
    }
    try:
        session_loop.query_knowledge_graph = _make_query_knowledge_graph(retrieval_context)
        session_loop.writer_agent = resolved_writer
        session_loop.auditor_agent = resolved_auditor
        # None -> session_loop falls back to the real, frozen
        # scripts.coordinator_agent.prepare_worker_brief. Never mocked.
        session_loop.coordinator_agent = None
        session_loop.chat_history_store = chat_history_store
        session_loop.mechanical_gate = resolved_gate

        raw_result = execute_session_loop_with_fallback(
            user_input=hypothesis,
            context={"retrieval_context": retrieval_context},
            session_id=session_id,
            reasoning_assessment=assessment,
        )
    finally:
        session_loop.query_knowledge_graph = saved["query_knowledge_graph"]
        session_loop.writer_agent = saved["writer_agent"]
        session_loop.auditor_agent = saved["auditor_agent"]
        session_loop.coordinator_agent = saved["coordinator_agent"]
        session_loop.chat_history_store = saved["chat_history_store"]
        session_loop.mechanical_gate = saved["mechanical_gate"]

    # ── 4. Normalize session_loop's result into this module's contract ─────
    if isinstance(raw_result, dict):
        session_result: dict[str, Any] = raw_result
        status = raw_result.get("status", "ESCALATED_TO_HUMAN")
    else:
        session_result = {"draft": raw_result}
        if isinstance(raw_result, str) and "INSUFFICIENT VARIABLES:" in raw_result:
            status = "INSUFFICIENT_VARIABLES"
        else:
            status = "ANSWER_SUCCESS"

    audit: Optional[dict] = None
    if isinstance(resolved_auditor, RecordingAuditor) and resolved_auditor.last_result is not None:
        audit = {"final_audit_result": resolved_auditor.last_result}

    return {
        "status": status,
        "assessment": assessment,
        "policy_action": policy_action,
        "session_result": session_result,
        "audit": audit,
        "trace_id": trace_id,
    }
