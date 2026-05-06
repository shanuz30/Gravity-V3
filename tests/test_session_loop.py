import unittest
from unittest.mock import MagicMock

import scripts.session_loop as session_loop
from scripts.session_loop import execute_session_loop_with_fallback


class TestSessionLoop(unittest.TestCase):

    def setUp(self):
        # Inject fresh mocks before each test to ensure a clean slate
        session_loop.query_knowledge_graph = MagicMock(
            return_value={"fact": "The sky is blue"}
        )
        session_loop.writer_agent = MagicMock()
        session_loop.auditor_agent = MagicMock()
        session_loop.coordinator_agent = None  # disabled for base tests

    def test_immediate_success(self):
        """Test that the loop exits immediately if the first draft is verified."""
        session_loop.writer_agent.generate.return_value = "The sky is blue."
        session_loop.auditor_agent.check.return_value = "TRUTH_VERIFIED"

        result = execute_session_loop_with_fallback("What color is the sky?", {})

        self.assertEqual(result, "The sky is blue.")
        session_loop.writer_agent.generate.assert_called_once()
        session_loop.auditor_agent.check.assert_called_once()
        session_loop.writer_agent.regenerate.assert_not_called()

    def test_success_after_one_retry(self):
        """Test that the loop successfully regenerates and verifies after one failure."""
        session_loop.writer_agent.generate.return_value = "The sky is green."
        session_loop.auditor_agent.check.side_effect = [
            "CONFLICT_FOUND",
            "TRUTH_VERIFIED",
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
            "CONFLICT_FOUND",
            "CONFLICT_FOUND",
            "CONFLICT_FOUND",
        ]
        session_loop.writer_agent.regenerate.return_value = "The sky is still green."

        result = execute_session_loop_with_fallback("What color is the sky?", {})

        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "ESCALATED_TO_HUMAN")
        self.assertEqual(result["last_draft"], "The sky is still green.")
        self.assertEqual(result["conflict_summary"], "CONFLICT_FOUND")
        self.assertEqual(session_loop.auditor_agent.check.call_count, 3)
        self.assertEqual(session_loop.writer_agent.regenerate.call_count, 3)


class TestSessionLoopWithCoordinator(unittest.TestCase):
    """Tests for the session loop when a CoordinatorAgent is active."""

    def setUp(self):
        from scripts.coordinator_agent import CoordinatorAgent

        self.coordinator = CoordinatorAgent()
        session_loop.coordinator_agent = self.coordinator
        session_loop.query_knowledge_graph = MagicMock(
            return_value={"sky": "blue"}
        )
        session_loop.writer_agent = MagicMock()
        session_loop.auditor_agent = MagicMock()

    def tearDown(self):
        session_loop.coordinator_agent = None

    def test_context_is_parsed_before_writing(self):
        """parse_context must be called with the provided context dict."""
        session_loop.writer_agent.generate.return_value = "The sky is blue."
        session_loop.auditor_agent.check.return_value = "[TRUTH_VERIFIED]"

        execute_session_loop_with_fallback(
            "What color is the sky?", {"sky": "blue"}
        )

        self.assertEqual(self.coordinator._session_context["sky"], "blue")

    def test_writer_receives_worker_brief(self):
        """Writer agent must be called with a WORKER_BRIEF dict, not raw truth_data."""
        session_loop.writer_agent.generate.return_value = "The sky is blue."
        session_loop.auditor_agent.check.return_value = "[TRUTH_VERIFIED]"

        # Query includes "blue" so no contradiction is triggered
        execute_session_loop_with_fallback(
            "What shade of blue is the sky?", {"sky": "blue"}
        )

        self.assertTrue(session_loop.writer_agent.generate.called)
        call_kwargs = session_loop.writer_agent.generate.call_args.kwargs
        self.assertIn("canonical_facts", call_kwargs["facts"])
        self.assertIn("task", call_kwargs["facts"])

    def test_clarification_required_on_contradiction(self):
        """Loop must short-circuit with CLARIFICATION_REQUIRED on contradiction."""
        # Session context says sky=blue; query references sky but not blue
        session_loop.coordinator_agent.parse_context({"sky": "blue"})

        result = execute_session_loop_with_fallback(
            "The sky is green, right?", {}
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "CLARIFICATION_REQUIRED")
        self.assertIn("worker_brief", result)
        # Writer must NOT be called when contradictions exist
        session_loop.writer_agent.generate.assert_not_called()

    def test_no_contradiction_proceeds_normally(self):
        """When there is no contradiction, the loop should proceed to writing."""
        session_loop.writer_agent.generate.return_value = "The sky is blue."
        session_loop.auditor_agent.check.return_value = "[TRUTH_VERIFIED]"

        result = execute_session_loop_with_fallback(
            "What shade of blue is the sky?", {"sky": "blue"}
        )

        self.assertEqual(result, "The sky is blue.")
        session_loop.writer_agent.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
