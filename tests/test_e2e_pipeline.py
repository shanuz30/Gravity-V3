"""
Black-box verification suite for antigravity/runtime.py::run_end_to_end.

This file exercises ONLY the public entry point:

    from antigravity.runtime import run_end_to_end

It never imports or inspects internals of antigravity.runtime (no
RecordingAuditor, no _DeterministicWriter), and never reaches into
scripts.session_loop's module-level globals. Everything is verified purely
through run_end_to_end's own return dict contract:

    {status, assessment, policy_action, session_result, audit, trace_id}

Only `anthropic`'s client (`messages.create`) is mocked, via MagicMock --
exactly the boundary tests/test_runtime.py already mocks. Every other
collaborator on the chain (Antigravity, GANLoop, CRAGDetector, LCVModule,
bridge, coordinator_agent, session_loop, MechanicalGate, TruthSentryAuditor)
runs its real code.

The mock-client text recipes below intentionally reuse the same proven
patterns prototyped in tests/test_runtime.py (same CORE CLAIM / CONFIDENCE /
JSON structure the real GANLoop parser expects) rather than reinventing them
from scratch -- see that file for the original recipes this was modeled on.
Behavioral overlap with test_runtime.py's white-box coverage is expected and
intentional: this suite is the black-box layer CI actually runs to confirm
the same guarantees hold through the public surface alone.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from scripts.chat_history import ChatHistoryStore

from antigravity.runtime import run_end_to_end

# ---------------------------------------------------------------------------
# Mock-client recipes (same proven text patterns as tests/test_runtime.py).
# ---------------------------------------------------------------------------

RETRIEVAL = (
    "the antigravity system converges to zero drift under gamma "
    "damping correction across many iterations"
)

DISC_TEXT = "FATAL FLAW: none found in this pass."


def _resp(text):
    r = MagicMock()
    r.content = [MagicMock(text=text)]
    return r


def _make_mock_client(gen_text, disc_text, conv_text):
    def side_effect(*args, **kwargs):
        system = kwargs.get("system", "")
        if "GENERATOR" in system:
            return _resp(gen_text)
        if "DISCRIMINATOR" in system:
            return _resp(disc_text)
        return _resp(conv_text)

    client = MagicMock()
    client.messages.create = MagicMock(side_effect=side_effect)
    return client


def safe_high_confidence_client(retrieval=RETRIEVAL, confidence=0.92):
    """SAFE alarm, high confidence, evidence present -> ANSWER."""
    gen_text = f"Some analysis.\nCORE CLAIM: {retrieval}"
    conv_text = (
        f"{retrieval}\n"
        f"CONFIDENCE: {int(confidence * 100)}% - strong\n"
        f'JSON: {{"confidence": {confidence}, "tier3_risk": false}}'
    )
    return _make_mock_client(gen_text, DISC_TEXT, conv_text)


def full_drift_client():
    """Generator claim wildly unrelated to the retrieval anchor -> the real
    CRAGDetector computes a genuine FULL_DRIFT alarm (not a canned value)."""
    topic = "bananas ripen quickly in warm humid climates under high ethylene exposure"
    gen_text = f"Some unrelated analysis.\nCORE CLAIM: {topic}"
    conv_text = f'{topic}\nCONFIDENCE: 95% - strong\nJSON: {{"confidence": 0.95, "tier3_risk": false}}'
    return _make_mock_client(gen_text, DISC_TEXT, conv_text)


def low_confidence_client():
    """SAFE alarm but confidence below the 0.40 floor -> ESCALATE."""
    return safe_high_confidence_client(confidence=0.15)


def mechanical_gate_reject_client():
    """Otherwise-ANSWER-worthy claim, but the core claim text itself trips
    the MechanicalGate's destructive-pattern blacklist (DROP TABLE)."""
    tainted = f"{RETRIEVAL} DROP TABLE users"
    gen_text = f"Some analysis.\nCORE CLAIM: {tainted}"
    conv_text = f'{tainted}\nCONFIDENCE: 92% - strong\nJSON: {{"confidence": 0.92, "tier3_risk": false}}'
    return _make_mock_client(gen_text, DISC_TEXT, conv_text)


class _TempChatHistoryStore:
    """Context manager for a tempfile-backed ChatHistoryStore, same pattern
    as tests/test_chat_history.py and tests/test_runtime.py."""

    def __enter__(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        os.unlink(tmp.name)  # let the store create it fresh
        self.path = tmp.name
        self.store = ChatHistoryStore(storage_path=self.path)
        return self.store

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.path):
            os.unlink(self.path)


class TestRunEndToEndBlackBox(unittest.TestCase):
    """Covers run_end_to_end's documented return contract through its public
    dict surface only."""

    # -- safe / high confidence -----------------------------------------

    def test_safe_high_confidence_yields_answer_success(self):
        client = safe_high_confidence_client()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertEqual(result["status"], "ANSWER_SUCCESS")
        self.assertEqual(result["policy_action"], "answer")
        self.assertIsNotNone(result["assessment"])
        self.assertIsNotNone(result["audit"])
        self.assertTrue(result["trace_id"])

    # -- FULL_DRIFT -------------------------------------------------------

    def test_full_drift_yields_blocked(self):
        client = full_drift_client()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["policy_action"], "block")
        # A real drift alarm, not a mechanical-gate rejection: no draft ever
        # reached the writer, so there is nothing for the auditor to have
        # verified.
        self.assertIsNone(result["audit"])

    # -- low confidence -----------------------------------------------------

    def test_low_confidence_yields_antigravity_escalation(self):
        client = low_confidence_client()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertEqual(result["status"], "ANTIGRAVITY_ESCALATION")
        self.assertEqual(result["policy_action"], "escalate")

    # -- mechanical-gate rejection -------------------------------------------

    def test_mechanical_gate_rejection_yields_blocked_with_gate_reason(self):
        client = mechanical_gate_reject_client()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertEqual(result["status"], "BLOCKED")
        # Distinct from a policy-level BLOCK: the coordinator recommended
        # ANSWER (assessment says so via policy_action) -- it's the
        # *mechanical gate*, not drift/confidence policy, that rejected the
        # draft. The gate-specific reason is what proves which layer fired.
        self.assertEqual(result["policy_action"], "answer")
        self.assertEqual(
            result["session_result"].get("reason"), "mechanical_gate_rejected"
        )

    # -- audit persistence ----------------------------------------------

    def test_audit_persistence_records_entry_with_final_status_on_success(self):
        with _TempChatHistoryStore() as store:
            client = safe_high_confidence_client()
            result = run_end_to_end(
                "some hypothesis",
                RETRIEVAL,
                anthropic_client=client,
                chat_history_store=store,
                session_id="e2e-sess-success",
            )
            self.assertEqual(result["status"], "ANSWER_SUCCESS")

            recent = store.get_recent(limit=1)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0].session_id, "e2e-sess-success")
            # session_loop persists the raw session_result (a plain verified
            # draft string in this recipe); chat_history's own status
            # resolution maps that to "SUCCESS" -- the same real outcome
            # this module reports as ANSWER_SUCCESS.
            self.assertEqual(recent[0].status, "SUCCESS")

    def test_audit_persistence_records_entry_with_final_status_on_block(self):
        with _TempChatHistoryStore() as store:
            client = full_drift_client()
            result = run_end_to_end(
                "some hypothesis",
                RETRIEVAL,
                anthropic_client=client,
                chat_history_store=store,
                session_id="e2e-sess-blocked",
            )
            self.assertEqual(result["status"], "BLOCKED")

            recent = store.get_recent(limit=1)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0].session_id, "e2e-sess-blocked")
            self.assertEqual(recent[0].status, "BLOCKED")

    def test_audit_persistence_records_entry_for_mechanical_gate_rejection(self):
        with _TempChatHistoryStore() as store:
            client = mechanical_gate_reject_client()
            result = run_end_to_end(
                "some hypothesis",
                RETRIEVAL,
                anthropic_client=client,
                chat_history_store=store,
                session_id="e2e-sess-gate",
            )
            self.assertEqual(result["status"], "BLOCKED")

            recent = store.get_recent(limit=1)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0].session_id, "e2e-sess-gate")
            self.assertEqual(recent[0].status, "BLOCKED")


if __name__ == "__main__":
    unittest.main()
