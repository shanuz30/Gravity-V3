"""
[TRUTH-SENTRY / AUDITOR AGENT]
Default state: skepticism.
Protects the integrity of the Knowledge Graph by identifying Fact Contamination in drafts.

Audit Protocol:
1. Falsification Check: Look for what is WRONG by comparing the draft against the
   provided context or [WORKER_BRIEF].
2. Logic Gate Verification: Ensure conditional constraints are respected.
3. Entity Integrity: Verify that colors, materials, owners, and locations match
   the source data exactly.

Output Format:
- Altered or ignored fact  -> [CONFLICT_FOUND]: {hallucination} vs {Graph Truth}
- Missed logic constraint  -> [LOGIC_ERROR]: {Description of the failure}
- 100% compliant draft    -> [TRUTH_VERIFIED]
"""


class TruthSentry:
    """
    Mechanical judge of accuracy.  Does not provide creative suggestions.
    Compares a draft strictly against the provided truth_data and surfaces any
    discrepancy with a structured output token.
    """

    def check(self, draft: str, truth_data: dict) -> str:
        """
        Run the full audit protocol against the supplied draft.

        Args:
            draft:      The prose or code produced by the Writer agent.
            truth_data: Ground-truth facts from the Knowledge Graph (may be a
                        plain dict or a WORKER_BRIEF dict with nested keys).

        Returns:
            One of:
              "[TRUTH_VERIFIED]"
              "[CONFLICT_FOUND]: <hallucination> vs <Graph Truth>"
              "[LOGIC_ERROR]: <description>"
        """
        # Flatten WORKER_BRIEF structure if necessary
        facts = self._extract_facts(truth_data)
        constraints = self._extract_constraints(truth_data)

        # Step 1 – Entity Integrity (Falsification Check)
        conflict = self._check_entity_integrity(draft, facts)
        if conflict:
            return f"[CONFLICT_FOUND]: {conflict}"

        # Step 2 – Logic Gate Verification
        logic_error = self._check_logic_constraints(draft, constraints)
        if logic_error:
            return f"[LOGIC_ERROR]: {logic_error}"

        return "[TRUTH_VERIFIED]"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_facts(self, truth_data: dict) -> dict:
        """Return the flat entity facts from either a raw dict or a WORKER_BRIEF."""
        if "canonical_facts" in truth_data:
            return truth_data["canonical_facts"]
        # Exclude constraint keys from plain dicts
        return {k: v for k, v in truth_data.items() if not k.startswith("constraint_")}

    def _extract_constraints(self, truth_data: dict) -> dict:
        """Return only the constraint entries from the truth data."""
        if "constraints" in truth_data:
            return truth_data["constraints"]
        return {k: v for k, v in truth_data.items() if k.startswith("constraint_")}

    def _check_entity_integrity(self, draft: str, facts: dict) -> str:
        """
        Verify that every string fact from the Knowledge Graph appears in the draft.
        Returns a conflict description string, or an empty string when clean.
        """
        draft_lower = draft.lower()
        for key, value in facts.items():
            if not isinstance(value, str):
                continue
            if value.lower() not in draft_lower:
                return (
                    f"Draft omits or contradicts '{value}' "
                    f"(Graph Truth for '{key}') vs what appears in the draft"
                )
        return ""

    def _check_logic_constraints(self, draft: str, constraints: dict) -> str:
        """
        Verify that every constraint value from the Knowledge Graph is honoured in
        the draft.  Returns a failure description, or an empty string when clean.
        """
        draft_lower = draft.lower()
        for key, value in constraints.items():
            if not isinstance(value, str):
                continue
            constraint_label = key.replace("constraint_", "").replace("_", " ")
            if value.lower() not in draft_lower:
                return (
                    f"Constraint '{constraint_label}' requires '{value}' "
                    f"but this condition is not reflected in the draft"
                )
        return ""
