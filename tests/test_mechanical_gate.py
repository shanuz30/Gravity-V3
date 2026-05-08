import unittest
import json
from scripts.mechanical_gate import MechanicalGate

class TestMechanicalGate(unittest.TestCase):
    def setUp(self):
        self.gate = MechanicalGate()

    def test_validate_command_safe(self):
        """Test that safe commands are allowed."""
        safe_commands = [
            "ls -l",
            "echo 'Hello, World!'",
            "cat README.md",
            "grep -r 'pattern' .",
            "python3 scripts/session_loop.py"
        ]
        for cmd in safe_commands:
            with self.subTest(cmd=cmd):
                self.assertTrue(self.gate.validate_command(cmd))

    def test_validate_command_blacklisted(self):
        """Test that blacklisted commands are blocked."""
        blacklisted = [
            "rm -rf /",
            "rm    -rf    /",
            "mkfs /dev/sdb1",
            "echo 'data' > /dev/sda",
            "DROP TABLE users;",
            "DELETE MATCH (n) RETURN n"
        ]
        for cmd in blacklisted:
            with self.subTest(cmd=cmd):
                self.assertFalse(self.gate.validate_command(cmd))

    def test_validate_command_case_insensitive(self):
        """Test that blacklist matching is case-insensitive."""
        blacklisted_mixed_case = [
            "RM -RF /",
            "MkFs /dev/sdb1",
            "drop table users",
            "delete match (n)"
        ]
        for cmd in blacklisted_mixed_case:
            with self.subTest(cmd=cmd):
                self.assertFalse(self.gate.validate_command(cmd))

    def test_cerebellum_translate_query_graph(self):
        """Test translation for query_graph action."""
        intent = json.dumps({"action": "query_graph", "query": "MATCH (n) RETURN n"})
        expected = "cypher: MATCH (n) RETURN n"
        self.assertEqual(self.gate.cerebellum_translate(intent), expected)

    def test_cerebellum_translate_other_action(self):
        """Test translation for other actions."""
        intent = json.dumps({"action": "summarize", "text": "Some text"})
        expected = "Translated execution for: summarize"
        self.assertEqual(self.gate.cerebellum_translate(intent), expected)

    def test_cerebellum_translate_invalid_json(self):
        """Test translation for invalid JSON."""
        invalid_json = "{'action': 'invalid'}"  # Single quotes are invalid in JSON
        self.assertEqual(self.gate.cerebellum_translate(invalid_json), "")

if __name__ == "__main__":
    unittest.main()
