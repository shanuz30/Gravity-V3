import unittest

from scripts.coordinator_agent import CoordinatorAgent


class TestCoordinatorAgentContextPreservation(unittest.TestCase):

    def setUp(self):
        self.agent = CoordinatorAgent()

    def test_parse_context_stores_entries(self):
        """Context provided to parse_context must be retrievable via build_worker_brief."""
        self.agent.parse_context({"sky": "blue", "grass": "green"})
        brief = self.agent.build_worker_brief("Describe the sky", {"sky": "blue"})
        self.assertEqual(brief["session_context"]["sky"], "blue")
        self.assertEqual(brief["session_context"]["grass"], "green")

    def test_parse_context_accumulates_across_calls(self):
        """Multiple calls to parse_context must merge, not replace."""
        self.agent.parse_context({"sky": "blue"})
        self.agent.parse_context({"grass": "green"})
        brief = self.agent.build_worker_brief("Describe nature", {})
        self.assertIn("sky", brief["session_context"])
        self.assertIn("grass", brief["session_context"])


class TestCoordinatorAgentWorkerBrief(unittest.TestCase):

    def setUp(self):
        self.agent = CoordinatorAgent()

    def test_brief_separates_facts_from_constraints(self):
        """canonical_facts must not include constraint_ keys; constraints must."""
        truth_data = {
            "material": "steel",
            "owner": "Alice",
            "constraint_time": "night",
        }
        brief = self.agent.build_worker_brief("Describe the part", truth_data)
        self.assertIn("material", brief["canonical_facts"])
        self.assertIn("owner", brief["canonical_facts"])
        self.assertNotIn("constraint_time", brief["canonical_facts"])
        self.assertIn("constraint_time", brief["constraints"])

    def test_brief_contains_task(self):
        """The task field must equal the user_input passed in."""
        brief = self.agent.build_worker_brief("Inspect valve #7", {})
        self.assertEqual(brief["task"], "Inspect valve #7")

    def test_brief_session_context_is_a_copy(self):
        """Mutating the returned brief must not affect internal state."""
        self.agent.parse_context({"sky": "blue"})
        brief = self.agent.build_worker_brief("query", {})
        brief["session_context"]["sky"] = "red"  # mutate the copy
        brief2 = self.agent.build_worker_brief("query", {})
        self.assertEqual(brief2["session_context"]["sky"], "blue")


class TestCoordinatorAgentTruthSeeking(unittest.TestCase):

    def setUp(self):
        self.agent = CoordinatorAgent()

    def test_no_contradiction_when_context_matches_input(self):
        """No contradiction when the query confirms the stored fact."""
        self.agent.parse_context({"sky": "blue"})
        result = self.agent.check_contradiction(
            "What shade of blue is the sky?", {"sky": "blue"}
        )
        self.assertFalse(result)

    def test_contradiction_when_context_fact_absent_from_input(self):
        """Contradiction when the query references the entity but not its value."""
        self.agent.parse_context({"sky": "blue"})
        result = self.agent.check_contradiction(
            "The sky is green, right?", {"sky": "blue"}
        )
        self.assertTrue(result)

    def test_no_contradiction_with_empty_context(self):
        """No contradiction when session context is empty."""
        result = self.agent.check_contradiction(
            "Tell me about anything", {"fact": "value"}
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
