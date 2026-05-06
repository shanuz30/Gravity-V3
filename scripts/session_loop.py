"""
[SESSION LOOP WITH FALLBACK]
Implements the writer-auditor feedback loop grounded by the knowledge graph.
On repeated conflict, escalates to human review instead of looping indefinitely.
"""

# These module-level references are replaced at runtime or during testing.
query_knowledge_graph = None
writer_agent = None
auditor_agent = None
coordinator_agent = None


def execute_session_loop_with_fallback(user_input: str, context: dict):
    """
    Generates a verified draft via a write-audit loop grounded in the knowledge graph.

    The CoordinatorAgent (when present) anchors the session: it parses the
    provided context for fact preservation, builds a [WORKER_BRIEF] from the
    knowledge-graph truth data, and performs a contradiction check before the
    Writer is invoked.  If a contradiction is detected the loop short-circuits
    and returns a CLARIFICATION_REQUIRED response instead of delegating to the
    Writer.

    Args:
        user_input: The user's query or instruction.
        context:    Ambient session context (entities, facts, constraints).

    Returns:
        A verified draft string on success, a CLARIFICATION_REQUIRED dict when
        the request contradicts the established context, or an ESCALATED_TO_HUMAN
        dict after max_attempts audit failures.
    """
    truth_data = query_knowledge_graph(user_input)

    if coordinator_agent is not None:
        # Context Preservation: anchor the session with the provided context.
        coordinator_agent.parse_context(context)

        # Truth Seeking: stop if the request contradicts known context.
        if coordinator_agent.check_contradiction(user_input, truth_data):
            worker_brief = coordinator_agent.build_worker_brief(user_input, truth_data)
            return {
                "status": "CLARIFICATION_REQUIRED",
                "reason": (
                    "Request contradicts established context. "
                    "Please resolve before proceeding."
                ),
                "worker_brief": worker_brief,
            }

        # Fact-First Analysis: build the [WORKER_BRIEF] for the Writer.
        worker_brief = coordinator_agent.build_worker_brief(user_input, truth_data)
        draft = writer_agent.generate(prompt=user_input, facts=worker_brief)
    else:
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
