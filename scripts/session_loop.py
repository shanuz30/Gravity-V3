"""
[SESSION LOOP WITH FALLBACK]
Implements the writer-auditor feedback loop grounded by the knowledge graph.
On repeated conflict, escalates to human review instead of looping indefinitely.
"""

from scripts.coordinator_agent import decide_final_action
from scripts.reasoning_assessment import RecommendedAction

# These module-level references are replaced at runtime or during testing.
query_knowledge_graph = None
writer_agent = None
auditor_agent = None
coordinator_agent = None

# Optional ChatHistoryStore instance. When set, every session result is persisted.
chat_history_store = None

# Optional MechanicalGate-like object (needs a .validate_command(str) -> bool
# method). When set, the final verified draft is checked before being
# returned. When None (the default), behavior is unchanged from before this
# gate existed.
mechanical_gate = None

_POLICY_SHORT_CIRCUIT_STATUS = {
    RecommendedAction.BLOCK: "BLOCKED",
    RecommendedAction.ESCALATE: "ANTIGRAVITY_ESCALATION",
    RecommendedAction.RETRIEVE: "RETRIEVAL_REQUIRED",
    RecommendedAction.CLARIFY: "CLARIFICATION_REQUIRED",
}


def execute_session_loop_with_fallback(
    user_input: str, context: dict, session_id: str = None, reasoning_assessment=None
):
    """
    Generates a verified draft via a write-audit loop grounded in the knowledge graph.

    Args:
        user_input: The user's query or instruction.
        context:    Ambient session context (passed through for future extensibility).
        reasoning_assessment: Optional scripts.reasoning_assessment.ReasoningAssessment
            (e.g. from antigravity.bridge). When None (the default), behavior is
            identical to before this parameter existed. When provided, Gravity-V3's
            coordinator (decide_final_action) decides the policy action first:
            BLOCK/ESCALATE/RETRIEVE/CLARIFY short-circuit before any retrieval or
            writing happens; ANSWER proceeds through the loop below exactly as
            usual, except the mechanical_gate (if set) now checks the final draft.

    Returns:
        A verified draft string on success, an escalation dict after max_attempts
        failures, or a policy short-circuit dict (status one of BLOCKED /
        ANTIGRAVITY_ESCALATION / RETRIEVAL_REQUIRED / CLARIFICATION_REQUIRED)
        when reasoning_assessment calls for one.
    """
    if reasoning_assessment is not None:
        decision = decide_final_action(reasoning_assessment)
        if decision.action != RecommendedAction.ANSWER:
            result = {
                "status": _POLICY_SHORT_CIRCUIT_STATUS[decision.action],
                "reason": decision.reason,
                "policy_action": decision.action.value,
            }
            _maybe_record(user_input, result, session_id)
            return result

    truth_data = query_knowledge_graph(user_input)

    worker_brief = _prepare_worker_brief(user_input=user_input, truth_data=truth_data, context=context)
    if "INSUFFICIENT VARIABLES:" in worker_brief:
        _maybe_record(user_input, worker_brief, session_id)
        return worker_brief

    draft = writer_agent.generate(prompt=worker_brief, facts=truth_data)

    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        audit_result = _call_auditor_check(draft=draft, truth_data=truth_data, worker_brief=worker_brief)

        if "TRUTH_VERIFIED" in audit_result:
            if mechanical_gate is not None and not mechanical_gate.validate_command(draft):
                blocked = {"status": "BLOCKED", "reason": "mechanical_gate_rejected", "draft": draft}
                _maybe_record(user_input, blocked, session_id)
                return blocked
            _maybe_record(user_input, draft, session_id)
            return draft

        # Conflict detected — regenerate and retry
        print(f"Attempt {attempts + 1}: Conflict detected. Retrying...")
        draft = writer_agent.regenerate(draft, audit_result)
        attempts += 1

    # STOP CONDITION: escalate to human after exhausting retries
    result = {
        "status": "ESCALATED_TO_HUMAN",
        "last_draft": draft,
        "conflict_summary": audit_result,
        "action_required": (
            "Please resolve the contradiction between the Graph and the Writer."
        ),
    }
    _maybe_record(user_input, result, session_id)
    return result


def _prepare_worker_brief(user_input: str, truth_data: dict, context: dict) -> str:
    if coordinator_agent is not None:
        return coordinator_agent.prepare_worker_brief(user_input, truth_data, context)

    from scripts.coordinator_agent import prepare_worker_brief

    return prepare_worker_brief(user_input, truth_data, context)


def _call_auditor_check(*, draft: str, truth_data: dict, worker_brief: str) -> str:
    try:
        return auditor_agent.check(draft, truth_data, worker_brief=worker_brief)
    except TypeError:
        try:
            return auditor_agent.check(draft, truth_data, worker_brief)
        except TypeError:
            return auditor_agent.check(draft, truth_data)


def _maybe_record(user_input: str, result, session_id) -> None:
    if chat_history_store is None:
        return
    from scripts.chat_history import record_session_result
    record_session_result(chat_history_store, user_input, result, session_id)
