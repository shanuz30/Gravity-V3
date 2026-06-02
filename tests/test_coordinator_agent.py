import unittest

from scripts.coordinator_agent import prepare_worker_brief


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


if __name__ == "__main__":
    unittest.main()

