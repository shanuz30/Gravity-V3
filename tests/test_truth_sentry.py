import unittest

from scripts.truth_sentry import TruthSentryAuditor


class TestTruthSentryAuditor(unittest.TestCase):
    def test_truth_verified(self):
        auditor = TruthSentryAuditor()
        truth_data = {"constraints": ["Time: Night"], "canonical_facts": ["The map is vellum."]}
        result = auditor.check("At night, the map is vellum.", truth_data, worker_brief=None)
        self.assertEqual(result, "[TRUTH_VERIFIED]")

    def test_logic_error(self):
        auditor = TruthSentryAuditor()
        truth_data = {"constraints": ["Time: Night"]}
        result = auditor.check("In daylight, proceed.", truth_data, worker_brief=None)
        self.assertTrue(result.startswith("[LOGIC_ERROR]:"))

    def test_conflict_found(self):
        auditor = TruthSentryAuditor()
        truth_data = {"canonical_facts": ["Owner: Alice"]}
        result = auditor.check("Owner: Bob", truth_data, worker_brief=None)
        self.assertTrue(result.startswith("[CONFLICT_FOUND]:"))


if __name__ == "__main__":
    unittest.main()

