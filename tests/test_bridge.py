"""
Tests for antigravity/bridge.py.

Skip-guarded: antigravity/* requires numpy/scipy/anthropic, which aren't
installed in every environment (including the sandbox this was written in).
These tests report `skipped` rather than failing when numpy is unavailable,
and are expected to pass once `pip install -r requirements.txt` succeeds.
"""

import unittest

try:
    import numpy as np  # noqa: F401

    from antigravity.core.types import AnchorState, GANSignal, CRAGSignal, LCVOutput
    from antigravity.orchestrator import PipelineState
    from antigravity.bridge import (
        AntigravityAssessmentError,
        make_producer_safe,
        to_reasoning_assessment,
        try_to_reasoning_assessment,
    )
    from scripts.reasoning_assessment import RecommendedAction

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


def _make_state(confidence, js, cos, damping_active, retrieval_id="anchor-1"):
    anchor = AnchorState(
        embedding=np.zeros(4), text="anchor text", timestamp=0.0, retrieval_id=retrieval_id
    )
    gan = GANSignal(
        confidence=confidence, core_claim="claim", fatal_flaw="", residual_truth="", rounds=1
    )
    crag = CRAGSignal(
        js_divergence=js, cosine_sim=cos, layer1_fired=js > 0.05, layer2_fired=cos < 0.70
    )
    lcv = LCVOutput(
        d=0.1, eps_base=0.1, eps_actual=0.1, damping_active=damping_active,
        correction_strength=0.5, converged=False, v_star=np.zeros(4),
    )
    return PipelineState(iteration=0, hypothesis="h", anchor=anchor, gan=gan, crag=crag, lcv=lcv)


@unittest.skipUnless(_DEPS_AVAILABLE, "antigravity dependencies (numpy) not installed")
class TestToReasoningAssessment(unittest.TestCase):
    def test_normal_confidence_maps_through(self):
        state = _make_state(confidence=0.9, js=0.0, cos=0.99, damping_active=False)
        a = to_reasoning_assessment(state, evidence_ids=("ev-1",))
        self.assertEqual(a.recommended_action, RecommendedAction.ANSWER)
        self.assertEqual(a.confidence, 0.9)
        self.assertEqual(a.alarm, "SAFE")
        self.assertFalse(a.damping_active)

    def test_high_crag_drift_maps_to_block(self):
        # js > 0.05 AND cos < 0.70 -> FULL_DRIFT
        state = _make_state(confidence=0.95, js=0.5, cos=0.1, damping_active=False)
        a = to_reasoning_assessment(state, evidence_ids=("ev-1",))
        self.assertEqual(a.alarm, "FULL_DRIFT")
        self.assertEqual(a.recommended_action, RecommendedAction.BLOCK)

    def test_lcv_damping_active_passes_through(self):
        state = _make_state(confidence=0.9, js=0.0, cos=0.99, damping_active=True)
        a = to_reasoning_assessment(state, evidence_ids=("ev-1",))
        self.assertTrue(a.damping_active)

    def test_low_confidence_maps_to_escalate(self):
        state = _make_state(confidence=0.1, js=0.0, cos=0.99, damping_active=False)
        a = to_reasoning_assessment(state, evidence_ids=("ev-1",))
        self.assertEqual(a.recommended_action, RecommendedAction.ESCALATE)

    def test_reasoning_trace_id_defaults_to_anchor_retrieval_id(self):
        state = _make_state(confidence=0.9, js=0.0, cos=0.99, damping_active=False, retrieval_id="abc123")
        a = to_reasoning_assessment(state, evidence_ids=("ev-1",))
        self.assertEqual(a.reasoning_trace_id, "abc123")

    def test_missing_signals_raises(self):
        state = PipelineState(iteration=0, hypothesis="h", anchor=None, gan=None, crag=None, lcv=None)
        with self.assertRaises(AntigravityAssessmentError):
            to_reasoning_assessment(state, evidence_ids=("ev-1",))


@unittest.skipUnless(_DEPS_AVAILABLE, "antigravity dependencies (numpy) not installed")
class TestTryToReasoningAssessment(unittest.TestCase):
    def test_malformed_state_degrades_to_none(self):
        state = PipelineState(iteration=0, hypothesis="h", anchor=None, gan=None, crag=None, lcv=None)
        self.assertIsNone(try_to_reasoning_assessment(state, evidence_ids=("ev-1",)))

    def test_valid_state_still_returns_assessment(self):
        state = _make_state(confidence=0.9, js=0.0, cos=0.99, damping_active=False)
        self.assertIsNotNone(try_to_reasoning_assessment(state, evidence_ids=("ev-1",)))


@unittest.skipUnless(_DEPS_AVAILABLE, "antigravity dependencies (numpy) not installed")
class TestMakeProducerSafe(unittest.TestCase):
    def test_simulated_timeout_degrades_to_none(self):
        def producer():
            raise TimeoutError("simulated antigravity API timeout")

        self.assertIsNone(make_producer_safe(producer, evidence_ids=("ev-1",)))

    def test_successful_producer_returns_assessment(self):
        def producer():
            return _make_state(confidence=0.9, js=0.0, cos=0.99, damping_active=False)

        result = make_producer_safe(producer, evidence_ids=("ev-1",))
        self.assertIsNotNone(result)
        self.assertEqual(result.recommended_action, RecommendedAction.ANSWER)


if __name__ == "__main__":
    unittest.main()
