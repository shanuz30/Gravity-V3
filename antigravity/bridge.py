"""
Bridge — the only file that imports across the Antigravity/Gravity-V3
boundary, and only in this direction: Antigravity (already dependent on
numpy/scipy/anthropic) depends on scripts.reasoning_assessment (pure
stdlib). scripts/ never imports antigravity.

Converts an Antigravity PipelineState into a ReasoningAssessment — a
deterministic, side-effect-free translation. No ID minting, no I/O, no
retries: reasoning_trace_id and evidence_ids are always caller-supplied.

Scope note on "timeout": no timeout mechanism exists in
antigravity/core/gan.py::GANLoop today, and this module does not add one
(that would mean modifying gan.py, out of scope). Instead, the conversion
step below is wrapped in a single broad exception handler in
try_to_reasoning_assessment — a real API timeout surfacing from whatever
produced the PipelineState degrades the same way a malformed/missing signal
does: to None, which the coordinator (scripts/coordinator_agent.py) treats
as BLOCK.
"""

from __future__ import annotations

from typing import Callable, Optional

from scripts.reasoning_assessment import ReasoningAssessment
from .orchestrator import PipelineState


class AntigravityAssessmentError(Exception):
    """Raised when a PipelineState doesn't carry enough signal to assess."""


def to_reasoning_assessment(
    state: PipelineState,
    *,
    evidence_ids: tuple[str, ...] = (),
    reasoning_trace_id: Optional[str] = None,
    confidence_floor: float = 0.40,
    risk_block_threshold: float = 0.90,
) -> ReasoningAssessment:
    """
    Build a ReasoningAssessment from an Antigravity PipelineState.

    Reuses state.gan.confidence, state.crag.alarm, and state.lcv.damping_active
    as-is — no reinterpretation of those signals happens here, only
    translation into the plain-value contract.

    reasoning_trace_id defaults to state.anchor.retrieval_id (already exists
    on every anchored PipelineState) when not supplied explicitly, so callers
    don't have to mint a new correlation id for something Antigravity already
    tracks.

    Raises AntigravityAssessmentError if state is missing the gan/crag/lcv
    signals a real pipeline run produces (i.e. the state is malformed).
    """
    if state.gan is None or state.crag is None or state.lcv is None:
        raise AntigravityAssessmentError(
            "PipelineState is missing gan/crag/lcv signals; "
            "run at least one Antigravity.run() iteration first."
        )

    trace_id = reasoning_trace_id
    if trace_id is None:
        if state.anchor is None or not state.anchor.retrieval_id:
            raise AntigravityAssessmentError(
                "No reasoning_trace_id supplied and state.anchor.retrieval_id "
                "is unavailable to fall back to."
            )
        trace_id = state.anchor.retrieval_id

    return ReasoningAssessment.create(
        alarm=state.crag.alarm,
        confidence=state.gan.confidence,
        damping_active=state.lcv.damping_active,
        evidence_ids=evidence_ids,
        reasoning_trace_id=trace_id,
        confidence_floor=confidence_floor,
        risk_block_threshold=risk_block_threshold,
    )


def try_to_reasoning_assessment(
    state: PipelineState,
    *,
    evidence_ids: tuple[str, ...] = (),
    reasoning_trace_id: Optional[str] = None,
    confidence_floor: float = 0.40,
    risk_block_threshold: float = 0.90,
) -> Optional[ReasoningAssessment]:
    """
    Same as to_reasoning_assessment, but degrades to None on any failure —
    malformed state, an unexpected exception, or (per the scope note above)
    a timeout surfacing from whatever produced `state` — instead of raising.

    The coordinator (scripts/coordinator_agent.py::decide_final_action)
    treats None as "malformed" and responds with BLOCK.
    """
    try:
        return to_reasoning_assessment(
            state,
            evidence_ids=evidence_ids,
            reasoning_trace_id=reasoning_trace_id,
            confidence_floor=confidence_floor,
            risk_block_threshold=risk_block_threshold,
        )
    except Exception:
        return None


def make_producer_safe(
    produce_state: Callable[[], PipelineState],
    *,
    evidence_ids: tuple[str, ...] = (),
    reasoning_trace_id: Optional[str] = None,
    confidence_floor: float = 0.40,
    risk_block_threshold: float = 0.90,
) -> Optional[ReasoningAssessment]:
    """
    Like try_to_reasoning_assessment, but also catches failures raised while
    *producing* the state (e.g. a network/API timeout from whatever ran the
    Antigravity pipeline), not just failures in the conversion itself.
    """
    try:
        state = produce_state()
    except Exception:
        return None
    return try_to_reasoning_assessment(
        state,
        evidence_ids=evidence_ids,
        reasoning_trace_id=reasoning_trace_id,
        confidence_floor=confidence_floor,
        risk_block_threshold=risk_block_threshold,
    )
