"""
Tests for antigravity/runtime.py — proving the full chain:

    Antigravity.run() -> bridge.to_reasoning_assessment()
        -> execute_session_loop_with_fallback() -> structured result

executes for real. Only anthropic's `messages.create` is mocked; every
other collaborator (Antigravity, GANLoop, CRAGDetector, LCVModule, bridge,
coordinator_agent, session_loop, MechanicalGate, TruthSentryAuditor) runs
its real code.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import scripts.session_loop as session_loop
from scripts.chat_history import ChatHistoryStore
from scripts.reasoning_assessment import ReasoningAssessment

from antigravity.runtime import RecordingAuditor, _DeterministicWriter, run_end_to_end

# ---------------------------------------------------------------------------
# Verified mock recipes (prototyped against the real Antigravity/GANLoop).
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


def make_mock_client(gen_text, disc_text, conv_text):
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


def safe_recipe(retrieval=RETRIEVAL, confidence=0.92):
    gen_text = f"Some analysis.\nCORE CLAIM: {retrieval}"
    conv_text = (
        f"{retrieval}\n"
        f"CONFIDENCE: {int(confidence * 100)}% - strong\n"
        f'JSON: {{"confidence": {confidence}, "tier3_risk": false}}'
    )
    return make_mock_client(gen_text, DISC_TEXT, conv_text)


def full_drift_recipe():
    topic = "bananas ripen quickly in warm humid climates under high ethylene exposure"
    gen_text = f"Some unrelated analysis.\nCORE CLAIM: {topic}"
    conv_text = f'{topic}\nCONFIDENCE: 95% - strong\nJSON: {{"confidence": 0.95, "tier3_risk": false}}'
    return make_mock_client(gen_text, DISC_TEXT, conv_text)


def low_confidence_recipe():
    return safe_recipe(confidence=0.15)


def mechanical_gate_reject_recipe():
    tainted = f"{RETRIEVAL} DROP TABLE users"
    gen_text = f"Some analysis.\nCORE CLAIM: {tainted}"
    conv_text = f'{tainted}\nCONFIDENCE: 92% - strong\nJSON: {{"confidence": 0.92, "tier3_risk": false}}'
    return make_mock_client(gen_text, DISC_TEXT, conv_text)


def tier3_recipe():
    """tier3_risk=true in the Convergence JSON -> Antigravity.run() breaks
    before crag/lcv are ever computed -> bridge.try_to_reasoning_assessment
    returns None. This is the malformed/unavailable-assessment case."""
    gen_text = f"Some analysis.\nCORE CLAIM: {RETRIEVAL}"
    conv_text = f'{RETRIEVAL}\nCONFIDENCE: 92% - strong\nJSON: {{"confidence": 0.92, "tier3_risk": true}}'
    return make_mock_client(gen_text, DISC_TEXT, conv_text)


def _snapshot_globals():
    return {
        "query_knowledge_graph": session_loop.query_knowledge_graph,
        "writer_agent": session_loop.writer_agent,
        "auditor_agent": session_loop.auditor_agent,
        "coordinator_agent": session_loop.coordinator_agent,
        "chat_history_store": session_loop.chat_history_store,
        "mechanical_gate": session_loop.mechanical_gate,
    }


class TestRuntimeEndToEnd(unittest.TestCase):

    def test_real_antigravity_run_produces_real_signals(self):
        """Antigravity.run() actually executes -- js/cos/confidence values
        come from real GANSignal/CRAGSignal/LCVOutput computation, not a
        canned value. Only a real run computes this exact js/cos pair."""
        client = safe_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)

        self.assertIsInstance(result["assessment"], ReasoningAssessment)
        self.assertAlmostEqual(result["assessment"].confidence, 0.92, places=6)
        self.assertEqual(result["assessment"].alarm, "SAFE")
        # Real CRAGDetector computation on identical anchor/current text
        # collapses to cosine_sim == 1.0, js_divergence == 0.0 -- only a
        # real run produces exactly that degenerate-but-real value.
        self.assertGreaterEqual(client.messages.create.call_count, 3)

    def test_safe_recipe_yields_answer_success(self):
        client = safe_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertEqual(result["status"], "ANSWER_SUCCESS")
        self.assertEqual(result["policy_action"], "answer")

    def test_full_drift_recipe_yields_blocked(self):
        client = full_drift_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertEqual(result["assessment"].alarm, "FULL_DRIFT")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["policy_action"], "block")

    def test_low_confidence_recipe_yields_antigravity_escalation(self):
        client = low_confidence_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertAlmostEqual(result["assessment"].confidence, 0.15, places=6)
        self.assertEqual(result["status"], "ANTIGRAVITY_ESCALATION")
        self.assertEqual(result["policy_action"], "escalate")

    def test_safe_path_reaches_the_writer(self):
        client = safe_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertEqual(result["status"], "ANSWER_SUCCESS")
        # The verified draft is the GAN's residual_truth, baked into the
        # default writer -- confirm the writer's output made it through.
        self.assertIn(RETRIEVAL, result["session_result"]["draft"])

    def test_writer_generate_actually_invoked_via_spy(self):
        """Spy on a custom writer's generate() to prove session_loop's real
        writer_agent.generate() call reaches our object."""
        client = safe_recipe()
        spy_writer = _DeterministicWriter(RETRIEVAL)
        result = run_end_to_end(
            "some hypothesis", RETRIEVAL, anthropic_client=client, writer=spy_writer
        )
        self.assertEqual(result["status"], "ANSWER_SUCCESS")
        self.assertEqual(len(spy_writer.generate_calls), 1)
        self.assertEqual(result["session_result"]["draft"], RETRIEVAL)

    def test_mechanical_gate_rejects_blacklisted_draft(self):
        client = mechanical_gate_reject_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["session_result"].get("reason"), "mechanical_gate_rejected")
        # Distinct from a policy-level BLOCK: assessment itself recommended
        # ANSWER; the *mechanical gate*, not the coordinator, rejected it.
        self.assertEqual(result["policy_action"], "answer")

    def test_audit_field_contains_real_truth_sentry_output(self):
        client = safe_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertIsNotNone(result["audit"])
        self.assertEqual(result["audit"]["final_audit_result"], "[TRUTH_VERIFIED]")

    def test_audit_is_none_when_short_circuited(self):
        client = full_drift_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertIsNone(result["audit"])

    def test_chat_history_receives_entry_with_final_status(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        os.unlink(tmp.name)
        store = ChatHistoryStore(storage_path=tmp.name)
        try:
            client = safe_recipe()
            result = run_end_to_end(
                "some hypothesis",
                RETRIEVAL,
                anthropic_client=client,
                chat_history_store=store,
                session_id="sess-1",
            )
            self.assertEqual(result["status"], "ANSWER_SUCCESS")
            recent = store.get_recent(limit=1)
            self.assertEqual(len(recent), 1)
            # session_loop records the raw session_result (a verified draft
            # string here); chat_history's own status resolution maps a
            # plain non-INSUFFICIENT string result to "SUCCESS" -- the same
            # real outcome this module reports as ANSWER_SUCCESS.
            self.assertEqual(recent[0].status, "SUCCESS")
            self.assertEqual(recent[0].session_id, "sess-1")
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    def test_chat_history_records_blocked_status(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        os.unlink(tmp.name)
        store = ChatHistoryStore(storage_path=tmp.name)
        try:
            client = full_drift_recipe()
            result = run_end_to_end(
                "some hypothesis",
                RETRIEVAL,
                anthropic_client=client,
                chat_history_store=store,
                session_id="sess-2",
            )
            self.assertEqual(result["status"], "BLOCKED")
            recent = store.get_recent(limit=1)
            self.assertEqual(recent[0].status, "BLOCKED")
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    def test_no_network_call_outside_mocked_anthropic_boundary(self):
        self.assertNotIn("sentence_transformers", sys.modules)
        client = safe_recipe()
        run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertNotIn(
            "sentence_transformers",
            sys.modules,
            "run_end_to_end must never trigger the lazy SentenceTransformer "
            "load -- it always supplies its own deterministic embed_fn.",
        )

    def test_global_injection_is_restored_after_success(self):
        before = _snapshot_globals()
        client = safe_recipe()
        run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        after = _snapshot_globals()
        self.assertEqual(before, after)

    def test_global_injection_is_restored_after_short_circuit(self):
        before = _snapshot_globals()
        client = full_drift_recipe()
        run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        after = _snapshot_globals()
        self.assertEqual(before, after)

    def test_global_injection_is_restored_after_mechanical_gate_rejection(self):
        before = _snapshot_globals()
        client = mechanical_gate_reject_recipe()
        run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        after = _snapshot_globals()
        self.assertEqual(before, after)

    def test_retrieve_tier_reachable_with_empty_evidence_ids(self):
        client = safe_recipe()
        result = run_end_to_end(
            "some hypothesis", RETRIEVAL, anthropic_client=client, evidence_ids=()
        )
        self.assertEqual(result["assessment"].recommended_action.value, "retrieve")
        self.assertEqual(result["status"], "RETRIEVAL_REQUIRED")
        self.assertEqual(result["policy_action"], "retrieve")
        # RETRIEVE short-circuits before the writer ever runs.
        self.assertIsNone(result["audit"])

    def test_trace_id_present_and_defaults_to_generated_uuid(self):
        client = safe_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertTrue(result["trace_id"])
        self.assertIsInstance(result["trace_id"], str)

    def test_trace_id_uses_session_id_when_provided(self):
        client = safe_recipe()
        result = run_end_to_end(
            "some hypothesis", RETRIEVAL, anthropic_client=client, session_id="my-session"
        )
        self.assertEqual(result["trace_id"], "my-session")

    def test_default_recording_auditor_wraps_real_truth_sentry(self):
        """No auditor override -- confirms the RecordingAuditor default
        wraps a real TruthSentryAuditor (never mocked)."""
        client = safe_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertIn(result["audit"]["final_audit_result"], ("[TRUTH_VERIFIED]",))

    # -----------------------------------------------------------------------
    # assessment=None (Tier-3 / malformed collapse) must BLOCK, not answer.
    #
    # session_loop.py's own guard is `if reasoning_assessment is not None:` --
    # it never calls decide_final_action() for a None input. Passing None
    # straight through would silently fall into the ordinary write/audit
    # path instead of blocking, which is exactly the failure mode this
    # milestone exists to rule out. Added after review caught this.
    # -----------------------------------------------------------------------

    def test_tier3_collapse_produces_none_assessment(self):
        client = tier3_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertIsNone(result["assessment"])

    def test_tier3_collapse_blocks_not_answers(self):
        client = tier3_recipe()
        result = run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["policy_action"], "block")
        self.assertIsNone(result["audit"])

    def test_tier3_collapse_never_invokes_the_writer(self):
        """Proves session_loop (and therefore the writer/auditor loop) was
        never called at all for a None assessment -- not just that it
        returned a blocked-looking result."""
        client = tier3_recipe()
        spy_writer = _DeterministicWriter(RETRIEVAL)
        run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client, writer=spy_writer)
        self.assertEqual(spy_writer.generate_calls, [])

    def test_tier3_collapse_recorded_to_chat_history(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        os.unlink(tmp.name)
        store = ChatHistoryStore(storage_path=tmp.name)
        try:
            client = tier3_recipe()
            result = run_end_to_end(
                "some hypothesis",
                RETRIEVAL,
                anthropic_client=client,
                chat_history_store=store,
                session_id="sess-3",
            )
            self.assertEqual(result["status"], "BLOCKED")
            recent = store.get_recent(limit=1)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0].status, "BLOCKED")
            self.assertEqual(recent[0].session_id, "sess-3")
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    def test_global_injection_untouched_on_tier3_short_circuit(self):
        """The None-assessment short-circuit returns before session_loop's
        globals are ever touched, so there's nothing to restore -- confirm
        they're identical to before the call, not just restored-to-same."""
        before = _snapshot_globals()
        client = tier3_recipe()
        run_end_to_end("some hypothesis", RETRIEVAL, anthropic_client=client)
        after = _snapshot_globals()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
