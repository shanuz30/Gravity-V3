"""
[REASONING ASSESSMENT — GRAVITY-V3 / ANTIGRAVITY BOUNDARY TYPE]

The single immutable contract Gravity-V3 accepts from the Antigravity
reasoning subsystem. Deliberately stdlib-only (`dataclasses`, `enum`) —
`scripts/` has no third-party dependencies today, and this module must not
change that. Antigravity (which already depends on numpy/scipy/anthropic)
is responsible for translating its own AnchorState / GANSignal / CRAGSignal /
LCVOutput into the plain values this module accepts — see
`antigravity/bridge.py`. The dependency direction is one-way: Antigravity
depends on this module; this module never imports Antigravity, and never
exposes any of its internal objects to Gravity-V3.

Gravity-V3 remains the policy and execution authority: `recommended_action`
here is Antigravity's *recommendation*, not a command. The final decision is
made downstream by `scripts/coordinator_agent.py::decide_final_action`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

VALID_ALARMS = ("SAFE", "LAYER1_WARN", "LAYER2_WARN", "FULL_DRIFT")

# risk_score blend weights and per-alarm severity. This formula is a new,
# deliberately simple policy-layer heuristic — it is NOT derived from the
# Wolfram/Lyapunov convergence proof in antigravity/core/lcv.py, and should
# not be read as implying a connection to it.
_RISK_CONFIDENCE_WEIGHT = 0.6
_RISK_ALARM_WEIGHT = 0.4
_ALARM_SEVERITY = {
    "SAFE": 0.0,
    "LAYER1_WARN": 0.5,
    "LAYER2_WARN": 0.5,
    "FULL_DRIFT": 1.0,
}


class RecommendedAction(str, Enum):
    """The bounded action vocabulary Antigravity can recommend."""
    ANSWER = "answer"
    RETRIEVE = "retrieve"
    CLARIFY = "clarify"
    ESCALATE = "escalate"
    BLOCK = "block"


_ESCALATING_ACTIONS = (RecommendedAction.ESCALATE, RecommendedAction.BLOCK)


@dataclass(frozen=True)
class ReasoningAssessment:
    """
    An immutable snapshot of Antigravity's drift/confidence state, reduced to
    primitives so `scripts/` can consume it without a hard dependency on
    numpy, scipy, or anthropic.

    Fields
    ------
    schema_version    : contract version string, e.g. "1.0".
    confidence         : [0.0, 1.0] — mirrors GANSignal.confidence.
    risk_score         : [0.0, 1.0] — derived; see _ALARM_SEVERITY blend above.
    alarm              : one of VALID_ALARMS — mirrors CRAGSignal.alarm.
    damping_active     : mirrors LCVOutput.damping_active.
    should_escalate    : True iff recommended_action is ESCALATE or BLOCK.
                          Not independently settable — enforced in __post_init__.
    recommended_action : Antigravity's recommendation. Gravity-V3 decides the
                          final action; see coordinator_agent.decide_final_action.
    evidence_ids       : retrieval/evidence identifiers backing this
                          assessment. Empty means "no evidence available".
    reasoning_trace_id : caller-supplied correlation id (e.g. an Antigravity
                          AnchorState.retrieval_id). Never minted internally —
                          keeps construction deterministic and side-effect-free.
    """
    schema_version: str
    confidence: float
    risk_score: float
    alarm: str
    damping_active: bool
    should_escalate: bool
    recommended_action: RecommendedAction
    evidence_ids: tuple[str, ...]
    reasoning_trace_id: str

    def __post_init__(self) -> None:
        if self.alarm not in VALID_ALARMS:
            raise ValueError(f"Unknown alarm '{self.alarm}'. Expected one of {VALID_ALARMS}.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0]. Got {self.confidence}.")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(f"risk_score must be in [0.0, 1.0]. Got {self.risk_score}.")
        if not self.reasoning_trace_id:
            raise ValueError("reasoning_trace_id is required (non-empty string).")
        expected_escalate = self.recommended_action in _ESCALATING_ACTIONS
        if self.should_escalate != expected_escalate:
            raise ValueError(
                f"should_escalate={self.should_escalate} is inconsistent with "
                f"recommended_action={self.recommended_action!r} "
                f"(expected should_escalate={expected_escalate})."
            )

    @classmethod
    def create(
        cls,
        *,
        alarm: str,
        confidence: float,
        damping_active: bool,
        evidence_ids: tuple[str, ...] = (),
        reasoning_trace_id: str,
        schema_version: str = "1.0",
        confidence_floor: float = 0.40,
        risk_block_threshold: float = 0.90,
    ) -> "ReasoningAssessment":
        """
        Build a fully self-consistent assessment from primitive signal values.

        confidence_floor defaults to 0.40, matching
        AntigravityConfig.gan_confidence_floor in antigravity/orchestrator.py.

        recommended_action decision rule, priority order, first match wins:
        1. alarm == "FULL_DRIFT" or risk_score >= risk_block_threshold -> BLOCK
        2. no evidence_ids                                             -> RETRIEVE
        3. confidence < confidence_floor                               -> ESCALATE
        4. alarm in (LAYER1_WARN, LAYER2_WARN)                         -> CLARIFY
        5. otherwise                                                   -> ANSWER

        This ordering is what makes contradictory signals deterministic: a
        real drift alarm always outranks a (possibly falsely) high confidence
        value, since tier 1 is checked before the confidence tier.
        """
        if alarm not in VALID_ALARMS:
            raise ValueError(f"Unknown alarm '{alarm}'. Expected one of {VALID_ALARMS}.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0]. Got {confidence}.")

        risk_score = _clamp01(
            _RISK_CONFIDENCE_WEIGHT * (1.0 - confidence)
            + _RISK_ALARM_WEIGHT * _ALARM_SEVERITY[alarm]
        )

        if alarm == "FULL_DRIFT" or risk_score >= risk_block_threshold:
            action = RecommendedAction.BLOCK
        elif not evidence_ids:
            action = RecommendedAction.RETRIEVE
        elif confidence < confidence_floor:
            action = RecommendedAction.ESCALATE
        elif alarm in ("LAYER1_WARN", "LAYER2_WARN"):
            action = RecommendedAction.CLARIFY
        else:
            action = RecommendedAction.ANSWER

        return cls(
            schema_version=schema_version,
            confidence=confidence,
            risk_score=risk_score,
            alarm=alarm,
            damping_active=damping_active,
            should_escalate=action in _ESCALATING_ACTIONS,
            recommended_action=action,
            evidence_ids=tuple(evidence_ids),
            reasoning_trace_id=reasoning_trace_id,
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class FakeAntigravityAdapter:
    """
    Zero-dependency stand-in for `antigravity.bridge`, for Gravity-V3-side
    tests (coordinator, session_loop) that need a ReasoningAssessment without
    ever importing antigravity or numpy. Mirrors the real adapter's call
    shape and delegates straight to ReasoningAssessment.create().
    """

    def assess(
        self,
        *,
        alarm: str,
        confidence: float,
        damping_active: bool,
        evidence_ids: tuple[str, ...] = (),
        reasoning_trace_id: str,
        **kwargs,
    ) -> ReasoningAssessment:
        return ReasoningAssessment.create(
            alarm=alarm,
            confidence=confidence,
            damping_active=damping_active,
            evidence_ids=evidence_ids,
            reasoning_trace_id=reasoning_trace_id,
            **kwargs,
        )
