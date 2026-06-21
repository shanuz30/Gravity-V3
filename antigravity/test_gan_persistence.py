"""
Tests for GAN loop memory.md / wiki.md persistence (Prompt 1).
Uses a fake Anthropic client — no API key or network access required.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from antigravity.core.gan import GANLoop


def _fake_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


CONV_HIGH_CONF = (
    "Para 1: survived.\nPara 2: uncertain.\nPara 3: next step.\n"
    "CONFIDENCE: 90% — strong\n"
    'JSON: {"confidence": 0.90, "tier3_risk": false}'
)

CONV_LOW_CONF = (
    "Para 1: survived.\nPara 2: uncertain.\nPara 3: next step.\n"
    "CONFIDENCE: 50% — weak\n"
    'JSON: {"confidence": 0.50, "tier3_risk": false}'
)

GEN_OUTPUT = "Some reasoning.\nCORE CLAIM: The system converges under gamma damping."
DISC_OUTPUT = "Attacks the assumption, finds a gap, however it fails.\nFATAL FLAW: Edge case at d=0."


class FakeClient:
    """Cycles through Generator -> Discriminator -> Convergence responses."""

    def __init__(self, conv_text: str):
        self._conv_text = conv_text
        self.calls = []
        self.messages = self

    def create(self, model, max_tokens, system, messages):
        self.calls.append({"system": system, "user": messages[0]["content"]})
        if "GENERATOR" in system:
            return _fake_response(GEN_OUTPUT)
        if "DISCRIMINATOR" in system:
            return _fake_response(DISC_OUTPUT)
        return _fake_response(self._conv_text)


class TestGANPersistence(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.state_dir = self._tmpdir.name
        self.memory_path = Path(self.state_dir) / "memory.md"
        self.wiki_path = Path(self.state_dir) / "wiki.md"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_loop(self, conv_text=CONV_HIGH_CONF, max_rounds=1):
        client = FakeClient(conv_text)
        gan = GANLoop(client, max_rounds=max_rounds, state_dir=self.state_dir)
        return gan, client

    def test_state_files_created_on_init(self):
        self._make_loop()
        self.assertTrue(self.memory_path.exists())
        self.assertTrue(self.wiki_path.exists())

    def test_memory_grows_by_one_entry_per_run(self):
        gan, _ = self._make_loop()
        gan.run("Hypothesis A", memory_context="seed")
        gan.run("Hypothesis B", memory_context="seed")
        gan.run("Hypothesis C", memory_context="seed")

        text = self.memory_path.read_text()
        self.assertEqual(text.count("---"), 3)
        self.assertIn("Hypothesis A", text)
        self.assertIn("Hypothesis B", text)
        self.assertIn("Hypothesis C", text)

    def test_wiki_unchanged_below_confidence_threshold(self):
        gan, _ = self._make_loop(conv_text=CONV_LOW_CONF)
        gan.run("Low confidence hypothesis", memory_context="seed")

        wiki_text = self.wiki_path.read_text()
        self.assertNotIn("Pattern", wiki_text)

    def test_wiki_updates_above_confidence_threshold(self):
        gan, _ = self._make_loop(conv_text=CONV_HIGH_CONF)
        gan.run("High confidence hypothesis", memory_context="seed")

        wiki_text = self.wiki_path.read_text()
        self.assertIn("Pattern [90%]", wiki_text)
        self.assertIn("The system converges under gamma damping.", wiki_text)

    def test_wiki_dedup_skips_duplicate_claim(self):
        gan, _ = self._make_loop(conv_text=CONV_HIGH_CONF)
        gan.run("First run", memory_context="seed")
        gan.run("Second run", memory_context="seed")

        wiki_text = self.wiki_path.read_text()
        self.assertEqual(wiki_text.count("Pattern ["), 1)

    def test_empty_memory_context_triggers_auto_load(self):
        gan, client = self._make_loop()
        gan.run("First hypothesis", memory_context="seed context")
        gan.run("Second hypothesis")  # no context passed -> should auto-load

        generator_calls = [c for c in client.calls if "GENERATOR" in c["system"]]
        second_call_input = generator_calls[1]["user"]  # round 0 of second run() call
        self.assertIn("First hypothesis"[:50], second_call_input)

    def test_default_state_dir_resolves_relative_to_module(self):
        client = FakeClient(CONV_HIGH_CONF)
        gan = GANLoop(client, max_rounds=1)
        expected = Path(__import__("antigravity.core.gan", fromlist=["__file__"]).__file__).resolve().parent.parent / "state"
        self.assertEqual(gan._memory_path.parent, expected)


if __name__ == "__main__":
    unittest.main()
