import unittest

from scripts.reasoning_assessment import (
    FakeAntigravityAdapter,
    ReasoningAssessment,
    RecommendedAction,
)


class TestReasoningAssessmentActions(unittest.TestCase):
    """One test per RecommendedAction value, exercised via .create()."""

    def test_full_drift_always_blocks(self):
        # Tier 1 (FULL_DRIFT) outranks everything else, even high confidence
        # and present evidence — this is the "contradictory signals" case.
        a = ReasoningAssessment.create(
            alarm="FULL_DRIFT", confidence=0.95, damping_active=True,
            evidence_ids=("ev-1",), reasoning_trace_id="trace-1",
        )
        self.assertEqual(a.recommended_action, RecommendedAction.BLOCK)
        self.assertTrue(a.should_escalate)

    def test_high_risk_score_blocks_even_without_full_drift(self):
        a = ReasoningAssessment.create(
            alarm="LAYER1_WARN", confidence=0.0, damping_active=False,
            evidence_ids=("ev-1",), reasoning_trace_id="trace-2",
            risk_block_threshold=0.5,
        )
        self.assertEqual(a.recommended_action, RecommendedAction.BLOCK)

    def test_missing_evidence_retrieves(self):
        a = ReasoningAssessment.create(
            alarm="SAFE", confidence=0.9, damping_active=False,
            evidence_ids=(), reasoning_trace_id="trace-3",
        )
        self.assertEqual(a.recommended_action, RecommendedAction.RETRIEVE)
        self.assertFalse(a.should_escalate)

    def test_low_confidence_escalates(self):
        a = ReasoningAssessment.create(
            alarm="SAFE", confidence=0.1, damping_active=False,
            evidence_ids=("ev-1",), reasoning_trace_id="trace-4",
        )
        self.assertEqual(a.recommended_action, RecommendedAction.ESCALATE)
        self.assertTrue(a.should_escalate)

    def test_warn_alarm_clarifies(self):
        a = ReasoningAssessment.create(
            alarm="LAYER2_WARN", confidence=0.8, damping_active=False,
            evidence_ids=("ev-1",), reasoning_trace_id="trace-5",
        )
        self.assertEqual(a.recommended_action, RecommendedAction.CLARIFY)
        self.assertFalse(a.should_escalate)

    def test_safe_high_confidence_with_evidence_answers(self):
        a = ReasoningAssessment.create(
            alarm="SAFE", confidence=0.9, damping_active=False,
            evidence_ids=("ev-1",), reasoning_trace_id="trace-6",
        )
        self.assertEqual(a.recommended_action, RecommendedAction.ANSWER)
        self.assertFalse(a.should_escalate)


class TestReasoningAssessmentValidation(unittest.TestCase):
    def test_confidence_below_zero_rejected(self):
        with self.assertRaises(ValueError):
            ReasoningAssessment.create(
                alarm="SAFE", confidence=-0.1, damping_active=False,
                evidence_ids=("ev-1",), reasoning_trace_id="t",
            )

    def test_confidence_above_one_rejected(self):
        with self.assertRaises(ValueError):
            ReasoningAssessment.create(
                alarm="SAFE", confidence=1.1, damping_active=False,
                evidence_ids=("ev-1",), reasoning_trace_id="t",
            )

    def test_unknown_alarm_rejected(self):
        with self.assertRaises(ValueError):
            ReasoningAssessment.create(
                alarm="NOT_REAL", confidence=0.5, damping_active=False,
                evidence_ids=("ev-1",), reasoning_trace_id="t",
            )

    def test_empty_reasoning_trace_id_rejected(self):
        with self.assertRaises(ValueError):
            ReasoningAssessment.create(
                alarm="SAFE", confidence=0.5, damping_active=False,
                evidence_ids=("ev-1",), reasoning_trace_id="",
            )

    def test_should_escalate_invariant_enforced_on_direct_construction(self):
        with self.assertRaises(ValueError):
            ReasoningAssessment(
                schema_version="1.0", confidence=0.9, risk_score=0.1,
                alarm="SAFE", damping_active=False,
                should_escalate=True,  # inconsistent with ANSWER
                recommended_action=RecommendedAction.ANSWER,
                evidence_ids=("ev-1",), reasoning_trace_id="t",
            )


class TestFakeAntigravityAdapter(unittest.TestCase):
    def test_fake_adapter_matches_direct_create(self):
        adapter = FakeAntigravityAdapter()
        via_adapter = adapter.assess(
            alarm="LAYER1_WARN", confidence=0.6, damping_active=True,
            evidence_ids=("ev-1", "ev-2"), reasoning_trace_id="trace-7",
        )
        via_direct = ReasoningAssessment.create(
            alarm="LAYER1_WARN", confidence=0.6, damping_active=True,
            evidence_ids=("ev-1", "ev-2"), reasoning_trace_id="trace-7",
        )
        self.assertEqual(via_adapter, via_direct)

    def test_fake_adapter_requires_no_third_party_imports(self):
        # Sanity: the module this test imports from must stay pure stdlib.
        import scripts.reasoning_assessment as mod

        with open(mod.__file__, "r", encoding="utf-8") as fh:
            source = fh.read()
        for pkg in ("numpy", "scipy", "anthropic"):
            self.assertNotIn(f"import {pkg}", source)


if __name__ == "__main__":
    unittest.main()
