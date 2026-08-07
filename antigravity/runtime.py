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

4. `assessment=None` (malformed PipelineState / Tier-3 collapse) is passed
   through unmodified as `reasoning_assessment=None` into
   `execute_session_loop_with_fallback`. Per session_loop's own documented
   behavior, a literal `None` there means "skip the policy short-circuit,
   behave exactly as before this parameter existed" — i.e. the ordinary
   write/audit loop still runs for real, and this module's `status` field
   reflects whatever that real loop actually produced (a verified draft, an
   ESCALATED_TO_HUMAN dict, or an INSUFFICIENT VARIABLES string). This
   module's `policy_action` field is still recorded as `"block"` in this
   case — mirroring `decide_final_action(None)`'s BLOCK mapping in the
   frozen `scripts/coordinator_agent.py` contract — but that is a *recorded*
   fact about what the assessment layer concluded, not a claim about what
   session_loop actually did with it, and not a re-decision (we never call
   `decide_final_action` ourselves). This module never inspects
   `state.alarm` or otherwise special-cases the Tier-3 path — the
   assessment=None collapse is accepted exactly as documented in the task
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

    # Recorded, not re-decided — see module docstring, note 4.
    policy_action = (
        assessment.recommended_action.value if assessment is not None else "block"
    )

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
