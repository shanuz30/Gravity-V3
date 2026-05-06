"""
[COORDINATOR AGENT]
Manages complex tasks by grounding them in established facts before delegation.

Operating Protocol:
1. Fact-First Analysis: Identifies load-bearing entities, rules, and constraints.
2. WORKER_BRIEF: Produces a structured block with Canonical Facts, Constraints, and Task.
3. Multi-Agent Orchestration: Prepares briefs for the Writer; does not write final output.
4. Truth Seeking: Stops when a request contradicts known context and requests clarification.
"""


class CoordinatorAgent:
    """
    Anchors the session by parsing context before processing any request.
    Produces a structured [WORKER_BRIEF] for downstream writer agents.
    """

    def __init__(self):
        self._session_context: dict = {}

    def parse_context(self, context: dict) -> None:
        """
        Context Preservation: Store the provided session context as the authoritative
        anchor for the current session.

        Args:
            context: Ambient session data (entities, facts, constraints, metadata).
        """
        self._session_context.update(context)

    def build_worker_brief(self, user_input: str, truth_data: dict) -> dict:
        """
        Fact-First Analysis: Identify load-bearing entities, rules, and constraints
        from the knowledge graph, then package them into a structured WORKER_BRIEF.

        Args:
            user_input: The user's raw query or instruction.
            truth_data: Facts retrieved from the knowledge graph.

        Returns:
            A dict representing the ### WORKER_BRIEF with canonical_facts,
            constraints, task, and session_context.
        """
        canonical_facts = {
            k: v for k, v in truth_data.items() if not k.startswith("constraint_")
        }
        constraints = {
            k: v for k, v in truth_data.items() if k.startswith("constraint_")
        }

        return {
            "canonical_facts": canonical_facts,
            "constraints": constraints,
            "task": user_input,
            "session_context": dict(self._session_context),
        }

    def check_contradiction(self, user_input: str, truth_data: dict) -> bool:
        """
        Truth Seeking: Determine whether the user's request contradicts any
        fact or constraint currently registered in the session context.

        A contradiction is detected when the session context asserts a value
        for an entity and the user's input asserts the opposite (e.g., the
        context states "sky: blue" but the query references the sky as green).

        Args:
            user_input: The user's raw query or instruction.
            truth_data: Facts retrieved from the knowledge graph.

        Returns:
            True if a contradiction is detected; False otherwise.
        """
        input_lower = user_input.lower()

        # Check session-context facts against explicit claims in the input
        for key, value in self._session_context.items():
            if not isinstance(value, str):
                continue
            value_lower = value.lower()
            # If the context fact appears to be negated or replaced in the query,
            # flag as a contradiction.  We use a lightweight heuristic: if the
            # context key (entity) is mentioned and its known value is *not* in
            # the query while another value-like token is present.
            if key.lower() in input_lower and value_lower not in input_lower:
                # Only flag when the user explicitly references the entity
                return True

        # Check truth_data constraints: if a constraint key is referenced in the
        # input but its required value is absent, report a contradiction.
        for key, value in truth_data.items():
            if not key.startswith("constraint_"):
                continue
            if isinstance(value, str) and value.lower() not in input_lower:
                # Constraint is present in graph but unaddressed in the request
                if key.lower().replace("constraint_", "") in input_lower:
                    return True

        return False
