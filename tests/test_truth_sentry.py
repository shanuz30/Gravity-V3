import unittest

from scripts.truth_sentry import TruthSentry


class TestTruthSentryEntityIntegrity(unittest.TestCase):

    def setUp(self):
        self.sentry = TruthSentry()

    def test_truth_verified_when_all_facts_present(self):
        """[TRUTH_VERIFIED] when every graph fact appears in the draft."""
        truth_data = {"color": "blue", "material": "steel"}
        draft = "The steel component is painted blue."
        result = self.sentry.check(draft, truth_data)
        self.assertEqual(result, "[TRUTH_VERIFIED]")

    def test_conflict_found_when_fact_missing_from_draft(self):
        """[CONFLICT_FOUND] when a fact from the graph is absent from the draft."""
        truth_data = {"color": "blue"}
        draft = "The component is painted red."
        result = self.sentry.check(draft, truth_data)
        self.assertIn("[CONFLICT_FOUND]", result)
        self.assertIn("blue", result)

    def test_conflict_found_for_owner_mismatch(self):
        """[CONFLICT_FOUND] when the owner is different in the draft."""
        truth_data = {"owner": "Alice"}
        draft = "The asset belongs to Bob."
        result = self.sentry.check(draft, truth_data)
        self.assertIn("[CONFLICT_FOUND]", result)

    def test_case_insensitive_fact_check(self):
        """Fact matching should be case-insensitive."""
        truth_data = {"color": "Blue"}
        draft = "The sky is blue today."
        result = self.sentry.check(draft, truth_data)
        self.assertEqual(result, "[TRUTH_VERIFIED]")


class TestTruthSentryLogicGates(unittest.TestCase):

    def setUp(self):
        self.sentry = TruthSentry()

    def test_logic_error_when_constraint_unmet(self):
        """[LOGIC_ERROR] when a constraint value is not reflected in the draft."""
        truth_data = {"constraint_time": "night"}
        draft = "The inscription is clearly visible in bright sunlight."
        result = self.sentry.check(draft, truth_data)
        self.assertIn("[LOGIC_ERROR]", result)
        self.assertIn("night", result)

    def test_truth_verified_when_constraint_met(self):
        """[TRUTH_VERIFIED] when the constraint value appears in the draft."""
        truth_data = {"constraint_time": "night"}
        draft = "The inscription is only legible at night under the stars."
        result = self.sentry.check(draft, truth_data)
        self.assertEqual(result, "[TRUTH_VERIFIED]")

    def test_entity_conflict_takes_priority_over_logic_error(self):
        """Entity conflicts are reported before logic errors."""
        truth_data = {"color": "blue", "constraint_time": "night"}
        draft = "The red object glows at night."
        result = self.sentry.check(draft, truth_data)
        self.assertIn("[CONFLICT_FOUND]", result)


class TestTruthSentryWorkerBriefInput(unittest.TestCase):
    """TruthSentry must handle WORKER_BRIEF dicts produced by CoordinatorAgent."""

    def setUp(self):
        self.sentry = TruthSentry()

    def test_accepts_worker_brief_format(self):
        """[TRUTH_VERIFIED] when a WORKER_BRIEF dict is supplied and draft is correct."""
        worker_brief = {
            "canonical_facts": {"material": "copper"},
            "constraints": {"constraint_temp": "below freezing"},
            "task": "Describe the pipe",
            "session_context": {},
        }
        draft = "The copper pipe operates efficiently below freezing temperatures."
        result = self.sentry.check(draft, worker_brief)
        self.assertEqual(result, "[TRUTH_VERIFIED]")

    def test_conflict_found_in_worker_brief_facts(self):
        """[CONFLICT_FOUND] when a canonical_fact in the WORKER_BRIEF is absent from draft."""
        worker_brief = {
            "canonical_facts": {"material": "copper"},
            "constraints": {},
            "task": "Describe the pipe",
            "session_context": {},
        }
        draft = "The steel pipe is strong."
        result = self.sentry.check(draft, worker_brief)
        self.assertIn("[CONFLICT_FOUND]", result)


if __name__ == "__main__":
    unittest.main()
