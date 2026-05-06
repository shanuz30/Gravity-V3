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


if __name__ == "__main__":
    unittest.main()
