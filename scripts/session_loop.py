"""
[SESSION LOOP WITH FALLBACK]
Implements the writer-auditor feedback loop grounded by the knowledge graph.
On repeated conflict, escalates to human review instead of looping indefinitely.
"""

# These module-level references are replaced at runtime or during testing.
query_knowledge_graph = None
writer_agent = None
auditor_agent = None


def execute_session_loop_with_fallback(user_input: str, context: dict):
    """
    Generates a verified draft via a write-audit loop grounded in the knowledge graph.

    Args:
        user_input: The user's query or instruction.
        context:    Ambient session context (passed through for future extensibility).

    Returns:
        A verified draft string on success, or an escalation dict after max_attempts failures.
    """
    truth_data = query_knowledge_graph(user_input)
    draft = writer_agent.generate(prompt=user_input, facts=truth_data)

    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        audit_result = auditor_agent.check(draft, truth_data)

        if "TRUTH_VERIFIED" in audit_result:
            return draft  # Optimal success path

        # Conflict detected — regenerate and retry
        print(f"Attempt {attempts + 1}: Conflict detected. Retrying...")
        draft = writer_agent.regenerate(draft, audit_result)
        attempts += 1

    # STOP CONDITION: escalate to human after exhausting retries
    return {
        "status": "ESCALATED_TO_HUMAN",
        "last_draft": draft,
        "conflict_summary": audit_result,
        "action_required": (
            "Please resolve the contradiction between the Graph and the Writer."
        ),
    }
