import unittest
from unittest.mock import MagicMock

import scripts.session_loop as session_loop
from scripts.session_loop import execute_session_loop_with_fallback
from scripts.reasoning_assessment import FakeAntigravityAdapter


class TestSessionLoop(unittest.TestCase):

    def setUp(self):
        # Inject fresh mocks before each test to ensure a clean slate
        session_loop.query_knowledge_graph = MagicMock(return_value={"fact": "The sky is blue"})
        session_loop.writer_agent = MagicMock()
        session_loop.auditor_agent = MagicMock()
        session_loop.coordinator_agent = MagicMock()
        session_loop.coordinator_agent.prepare_worker_brief.return_value = (
            "### WORKER_BRIEF\n\n"
            "- **Canonical Facts:**\n"
            "  - fact: The sky is blue\n"
            "- **Constraints:**\n"
            "  - (none provided)\n"
            "- **The Task (100 words):**\n"
            "  Implement the user request exactly. Verify every statement against the facts and constraints before finalizing.\n"
        )

    def test_immediate_success(self):
        """Test that the loop exits immediately if the first draft is verified."""
        session_loop.writer_agent.generate.return_value = "The sky is blue."
        session_loop.auditor_agent.check.return_value = "[TRUTH_VERIFIED]"

        result = execute_session_loop_with_fallback("What color is the sky?", {})

        self.assertEqual(result, "The sky is blue.")
        session_loop.coordinator_agent.prepare_worker_brief.assert_called_once_with(
            "What color is the sky?", {"fact": "The sky is blue"}, {}
        )
        session_loop.writer_agent.generate.assert_called_once_with(
            prompt=session_loop.coordinator_agent.prepare_worker_brief.return_value,
            facts={"fact": "The sky is blue"},
        )
        session_loop.auditor_agent.check.assert_called_once_with(
            "The sky is blue.",
            {"fact": "The sky is blue"},
            worker_brief=session_loop.coordinator_agent.prepare_worker_brief.return_value,
        )
        session_loop.writer_agent.regenerate.assert_not_called()

    def test_success_after_one_retry(self):
        """Test that the loop successfully regenerates and verifies after one failure."""
        session_loop.writer_agent.generate.return_value = "The sky is green."
        session_loop.auditor_agent.check.side_effect = [
            "[CONFLICT_FOUND]: sky green vs graph truth sky blue",
            "[TRUTH_VERIFIED]",
        ]
        session_loop.writer_agent.regenerate.return_value = "The sky is blue."

        result = execute_session_loop_with_fallback("What color is the sky?", {})

        self.assertEqual(result, "The sky is blue.")
        self.assertEqual(session_loop.auditor_agent.check.call_count, 2)
        session_loop.writer_agent.regenerate.assert_called_once()

    def test_fallback_escalation(self):
        """Test that the fallback mechanism triggers after max attempts."""
        session_loop.writer_agent.generate.return_value = "The sky is green."
        session_loop.auditor_agent.check.side_effect = [
            "[CONFLICT_FOUND]: sky green vs graph truth sky blue",
            "[CONFLICT_FOUND]: sky green vs graph truth sky blue",
            "[CONFLICT_FOUND]: sky green vs graph truth sky blue",
        ]
        session_loop.writer_agent.regenerate.return_value = "The sky is still green."

        result = execute_session_loop_with_fallback("What color is the sky?", {})

        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "ESCALATED_TO_HUMAN")
        self.assertEqual(result["last_draft"], "The sky is still green.")
        self.assertEqual(result["conflict_summary"], "[CONFLICT_FOUND]: sky green vs graph truth sky blue")
        self.assertEqual(session_loop.auditor_agent.check.call_count, 3)
        self.assertEqual(session_loop.writer_agent.regenerate.call_count, 3)

    def test_insufficient_variables_short_circuits(self):
        session_loop.coordinator_agent.prepare_worker_brief.return_value = (
            "### WORKER_BRIEF\n- INSUFFICIENT VARIABLES: Missing Time.\n"
        )

        result = execute_session_loop_with_fallback("Do the thing", {})

        self.assertIn("INSUFFICIENT VARIABLES: Missing Time.", result)
        session_loop.writer_agent.generate.assert_not_called()
        session_loop.auditor_agent.check.assert_not_called()


class TestSessionLoopBackwardCompatibility(unittest.TestCase):
    """Proves the new reasoning_assessment/mechanical_gate params change
    nothing when omitted -- the two ways of calling the old signature must
    produce byte-for-byte identical results."""

    def setUp(self):
        session_loop.query_knowledge_graph = MagicMock(return_value={"fact": "The sky is blue"})
        session_loop.writer_agent = MagicMock()
        session_loop.auditor_agent = MagicMock()
        session_loop.coordinator_agent = MagicMock()
        session_loop.coordinator_agent.prepare_worker_brief.return_value = "### WORKER_BRIEF\n"
        session_loop.writer_agent.generate.return_value = "The sky is blue."
        session_loop.auditor_agent.check.return_value = "[TRUTH_VERIFIED]"

    def tearDown(self):
        session_loop.mechanical_gate = None

    def test_omitting_kwarg_matches_explicit_none(self):
        result_omitted = execute_session_loop_with_fallback("What color is the sky?", {})
        result_explicit_none = execute_session_loop_with_fallback(
            "What color is the sky?", {}, reasoning_assessment=None
        )
        self.assertEqual(result_omitted, result_explicit_none)
        self.assertEqual(result_omitted, "The sky is blue.")


class TestSessionLoopWithReasoningAssessment(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAntigravityAdapter()
        session_loop.query_knowledge_graph = MagicMock(return_value={"fact": "The sky is blue"})
        session_loop.writer_agent = MagicMock()
        session_loop.auditor_agent = MagicMock()
        session_loop.coordinator_agent = MagicMock()
        session_loop.coordinator_agent.prepare_worker_brief.return_value = "### WORKER_BRIEF\n"
        session_loop.writer_agent.generate.return_value = "The sky is blue."
        session_loop.auditor_agent.check.return_value = "[TRUTH_VERIFIED]"

    def tearDown(self):
        session_loop.mechanical_gate = None

    def _assess(self, **overrides):
        params = dict(
            alarm="SAFE", confidence=0.9, damping_active=False,
            evidence_ids=("ev-1",), reasoning_trace_id="t",
        )
        params.update(overrides)
        return self.adapter.assess(**params)

    def test_block_short_circuits_before_retrieval_or_writing(self):
        a = self._assess(alarm="FULL_DRIFT")
        result = execute_session_loop_with_fallback("q", {}, reasoning_assessment=a)
        self.assertEqual(result["status"], "BLOCKED")
        session_loop.query_knowledge_graph.assert_not_called()
        session_loop.writer_agent.generate.assert_not_called()

    def test_escalate_short_circuits(self):
        a = self._assess(confidence=0.1)
        result = execute_session_loop_with_fallback("q", {}, reasoning_assessment=a)
        self.assertEqual(result["status"], "ANTIGRAVITY_ESCALATION")
        session_loop.writer_agent.generate.assert_not_called()

    def test_retrieve_short_circuits(self):
        a = self._assess(evidence_ids=())
        result = execute_session_loop_with_fallback("q", {}, reasoning_assessment=a)
        self.assertEqual(result["status"], "RETRIEVAL_REQUIRED")
        session_loop.writer_agent.generate.assert_not_called()

    def test_clarify_short_circuits(self):
        a = self._assess(alarm="LAYER2_WARN")
        result = execute_session_loop_with_fallback("q", {}, reasoning_assessment=a)
        self.assertEqual(result["status"], "CLARIFICATION_REQUIRED")
        session_loop.writer_agent.generate.assert_not_called()

    def test_answer_proceeds_through_normal_loop(self):
        a = self._assess()
        result = execute_session_loop_with_fallback("q", {}, reasoning_assessment=a)
        self.assertEqual(result, "The sky is blue.")
        session_loop.writer_agent.generate.assert_called_once()

    def test_answer_with_mechanical_gate_allowing_returns_draft(self):
        session_loop.mechanical_gate = MagicMock()
        session_loop.mechanical_gate.validate_command.return_value = True
        a = self._assess()
        result = execute_session_loop_with_fallback("q", {}, reasoning_assessment=a)
        self.assertEqual(result, "The sky is blue.")

    def test_answer_with_mechanical_gate_rejecting_returns_blocked(self):
        session_loop.mechanical_gate = MagicMock()
        session_loop.mechanical_gate.validate_command.return_value = False
        a = self._assess()
        result = execute_session_loop_with_fallback("q", {}, reasoning_assessment=a)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["reason"], "mechanical_gate_rejected")

    def test_audit_log_records_policy_short_circuit(self):
        session_loop.chat_history_store = MagicMock()
        a = self._assess(alarm="FULL_DRIFT")
        execute_session_loop_with_fallback("q", {}, session_id="s1", reasoning_assessment=a)
        session_loop.chat_history_store.record.assert_called_once()
        recorded_entry = session_loop.chat_history_store.record.call_args[0][0]
        self.assertEqual(recorded_entry.status, "BLOCKED")
        self.assertEqual(recorded_entry.session_id, "s1")
        session_loop.chat_history_store = None


if __name__ == "__main__":
    unittest.main()
