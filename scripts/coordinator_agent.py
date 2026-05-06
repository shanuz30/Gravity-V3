"""
[COORDINATOR AGENT]
Builds a fact-first WORKER_BRIEF from a knowledge graph payload and user request.
"""

from __future__ import annotations

from typing import Any, Iterable


def prepare_worker_brief(user_input: str, truth_data: dict[str, Any], context: dict[str, Any] | None = None) -> str:
    """
    Returns a markdown `### WORKER_BRIEF` block containing canonical facts, constraints,
    and an exact 100-word task scope for a sub-agent.

    If variables required by the knowledge graph are missing from the provided context,
    returns a brief containing an `INSUFFICIENT VARIABLES: ...` line and no task scope.
    """
    context = context or {}

    canonical_facts = _extract_canonical_facts(truth_data)
    constraints = _extract_constraints(truth_data)

    missing_variables = _missing_required_variables(truth_data, context)

    lines: list[str] = ["### WORKER_BRIEF", "", "- **Canonical Facts:**"]
    if canonical_facts:
        lines.extend([f"  - {fact}" for fact in canonical_facts])
    else:
        lines.append("  - (none provided)")

    lines.append("- **Constraints:**")
    if constraints:
        lines.extend([f"  - {constraint}" for constraint in constraints])
    else:
        lines.append("  - (none provided)")

    if missing_variables:
        missing_list = ", ".join(missing_variables)
        lines.append(f"- INSUFFICIENT VARIABLES: Missing {missing_list}.")
        return "\n".join(lines).rstrip() + "\n"

    task = _make_task_scope_100_words(user_input=user_input, canonical_facts=canonical_facts, constraints=constraints)
    lines.append("- **The Task (100 words):**")
    lines.append(f"  {task}")
    return "\n".join(lines).rstrip() + "\n"


def _extract_canonical_facts(truth_data: dict[str, Any]) -> list[str]:
    for key in ("canonical_facts", "facts"):
        if key in truth_data:
            return _stringify_items(truth_data.get(key))

    # Fall back to a stable, human-readable representation for small dict payloads.
    if isinstance(truth_data, dict) and truth_data:
        return [f"{k}: {truth_data[k]}" for k in sorted(truth_data.keys())]
    return []


def _extract_constraints(truth_data: dict[str, Any]) -> list[str]:
    for key in ("load_bearing_constraints", "constraints"):
        if key in truth_data:
            return _stringify_items(truth_data.get(key))
    return []


def _stringify_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [f"{k}: {value[k]}" for k in sorted(value.keys())]
    if isinstance(value, Iterable):
        items: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, dict):
                if "id" in item and len(item) == 1:
                    items.append(f"ID: {item['id']}")
                else:
                    parts = [f"{k}={item[k]}" for k in sorted(item.keys())]
                    items.append(", ".join(parts))
            else:
                items.append(str(item))
        return items
    return [str(value)]


def _missing_required_variables(truth_data: dict[str, Any], context: dict[str, Any]) -> list[str]:
    required = truth_data.get("required_variables")
    missing = truth_data.get("missing_variables")

    if isinstance(missing, list) and missing:
        return [str(v) for v in missing]

    if not isinstance(required, list) or not required:
        return []

    provided = context.get("variables") or context.get("provided_variables") or {}
    if not isinstance(provided, dict):
        provided = {}

    missing_vars = [str(v) for v in required if not provided.get(str(v))]
    return missing_vars


def _make_task_scope_100_words(user_input: str, canonical_facts: list[str], constraints: list[str]) -> str:
    # Deterministic base scope. We intentionally avoid creative elaboration.
    base = (
        "Implement the user request exactly: "
        + user_input.strip()
        + " Use only the Canonical Facts and obey all Constraints. "
        "Do not invent IDs, materials, owners, colors, locations, or numbers. "
        "If any required detail is missing or ambiguous, ask for clarification instead of guessing. "
        "Keep claims grounded, verifiable, and consistent with the brief. "
        "Return only the requested deliverable, with no extra commentary."
    )

    # Lightly bias the scope to remind the writer the facts/constraints exist, without copying them verbatim.
    if canonical_facts:
        base += " Treat the listed facts as authoritative."
    if constraints:
        base += " Treat the listed constraints as gates that must not be violated."

    return _enforce_exact_word_count(base, 100)


def _enforce_exact_word_count(text: str, target_words: int) -> str:
    words = [w for w in text.split() if w]
    if len(words) > target_words:
        return " ".join(words[:target_words])

    pad_words = [
        "Verify",
        "every",
        "statement",
        "against",
        "the",
        "facts",
        "and",
        "constraints",
        "before",
        "finalizing",
    ]
    i = 0
    while len(words) < target_words:
        words.append(pad_words[i % len(pad_words)])
        i += 1
    return " ".join(words[:target_words])

