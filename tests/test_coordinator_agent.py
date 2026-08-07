import unittest

from scripts.coordinator_agent import PolicyDecision, decide_final_action, prepare_worker_brief
from scripts.reasoning_assessment import FakeAntigravityAdapter, RecommendedAction


class TestCoordinatorAgent(unittest.TestCase):
    def test_worker_brief_contains_sections(self):
        truth_data = {
            "canonical_facts": ["ID: ITEM_MAP_001", "Material: Vellum", "Owner: Alice"],
            "constraints": ["Time: Night", "Requirement: Moonlight"],
        }

        brief = prepare_worker_brief("Summarize the map.", truth_data, context={})

        self.assertIn("### WORKER_BRIEF", brief)
        self.assertIn("**Canonical Facts:**", brief)
        self.assertIn("**Constraints:**", brief)
        self.assertIn("**The Task (100 words):**", brief)

        # Ensure exact 100-word task scope.
        task_line = [ln for ln in brief.splitlines() if ln.strip().startswith("Implement the user request exactly:")][0]
        self.assertEqual(len(task_line.split()), 100)

    def test_insufficient_variables(self):
        truth_data = {
            "canonical_facts": ["ID: ITEM_MAP_001"],
            "constraints": ["Time: Night"],
            "required_variables": ["Time"],
        }
        brief = prepare_worker_brief("Do it.", truth_data, context={"variables": {}})

        self.assertIn("### WORKER_BRIEF", brief)
        self.assertIn("INSUFFICIENT VARIABLES: Missing Time.", brief)
        self.assertNotIn("**The Task (100 words):**", brief)


class TestDecideFinalAction(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAntigravityAdapter()

    def test_none_assessment_blocks(self):
        decision = decide_final_action(None)
        self.assertIsInstance(decision, PolicyDecision)
        self.assertEqual(decision.action, RecommendedAction.BLOCK)
        self.assertEqual(decision.reason, "malformed_or_unavailable_assessment")
        self.assertIsNone(decision.assessment)

    def test_answer_passthrough(self):
        a = self.adapter.assess(
            alarm="SAFE", confidence=0.9, damping_active=False,
            evidence_ids=("ev-1",), reasoning_trace_id="t",
        )
        decision = decide_final_action(a)
        self.assertEqual(decision.action, RecommendedAction.ANSWER)
        self.assertIs(decision.assessment, a)

    def test_retrieve_passthrough(self):
        a = self.adapter.assess(
            alarm="SAFE", confidence=0.9, damping_active=False,
            evidence_ids=(), reasoning_trace_id="t",
        )
        self.assertEqual(decide_final_action(a).action, RecommendedAction.RETRIEVE)

    def test_clarify_passthrough(self):
        a = self.adapter.assess(
            alarm="LAYER1_WARN", confidence=0.9, damping_active=False,
            evidence_ids=("ev-1",), reasoning_trace_id="t",
        )
        self.assertEqual(decide_final_action(a).action, RecommendedAction.CLARIFY)

    def test_escalate_passthrough(self):
        a = self.adapter.assess(
            alarm="SAFE", confidence=0.1, damping_active=False,
            evidence_ids=("ev-1",), reasoning_trace_id="t",
        )
        self.assertEqual(decide_final_action(a).action, RecommendedAction.ESCALATE)

    def test_block_passthrough(self):
        a = self.adapter.assess(
            alarm="FULL_DRIFT", confidence=0.9, damping_active=False,
            evidence_ids=("ev-1",), reasoning_trace_id="t",
        )
        self.assertEqual(decide_final_action(a).action, RecommendedAction.BLOCK)


if __name__ == "__main__":
    unittest.main()

