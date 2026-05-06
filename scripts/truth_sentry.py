"""
[TRUTH-SENTRY AUDITOR]
Mechanical, falsification-first draft checker against a knowledge graph payload
and/or a WORKER_BRIEF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TruthSentryAuditor:
    def check(self, draft: str, truth_data: dict[str, Any], worker_brief: str | None = None) -> str:
        draft_text = draft or ""

        constraint_error = _check_constraints(draft_text, truth_data)
        if constraint_error:
            return f"[LOGIC_ERROR]: {constraint_error}"

        conflict_error = _check_canonical_facts(draft_text, truth_data, worker_brief)
        if conflict_error:
            return f"[CONFLICT_FOUND]: {conflict_error}"

        return "[TRUTH_VERIFIED]"


def _check_constraints(draft: str, truth_data: dict[str, Any]) -> str | None:
    constraints = truth_data.get("constraints") or truth_data.get("load_bearing_constraints") or []
    for constraint in _iter_items(constraints):
        if isinstance(constraint, dict):
            must_include = constraint.get("must_include") or []
            for term in _iter_items(must_include):
                term_s = str(term)
                if term_s and term_s.lower() not in draft.lower():
                    return f"Missing required term '{term_s}' vs constraint '{constraint}'."
            must_not_include = constraint.get("must_not_include") or []
            for term in _iter_items(must_not_include):
                term_s = str(term)
                if term_s and term_s.lower() in draft.lower():
                    return f"Found forbidden term '{term_s}' vs constraint '{constraint}'."
        else:
            # Heuristic: for constraints like "Time: Night", require the RHS token.
            constraint_s = str(constraint).strip()
            rhs = _rhs_token(constraint_s)
            if rhs and rhs.lower() not in draft.lower():
                return f"Missing required term '{rhs}' vs constraint '{constraint_s}'."
    return None


def _check_canonical_facts(draft: str, truth_data: dict[str, Any], worker_brief: str | None) -> str | None:
    canonical = truth_data.get("canonical_facts") or truth_data.get("facts") or []

    # If facts are expressed as strings, treat them as required anchors.
    str_facts = [str(f).strip() for f in _iter_items(canonical) if isinstance(f, str) and str(f).strip()]
    for fact in str_facts:
        if fact.lower() not in draft.lower():
            graph_truth = fact
            return f"draft missing '{fact}' vs graph truth '{graph_truth}'"

    # If a WORKER_BRIEF is available, ensure the draft does not contradict the brief's explicit IDs.
    if worker_brief:
        ids = _extract_ids(worker_brief)
        for identifier in ids:
            if identifier.lower() in draft.lower():
                continue
            # Only flag if draft appears to discuss IDs at all.
            if "id" in draft.lower():
                return f"draft omitted required ID '{identifier}' vs brief '{identifier}'"

    return None


def _rhs_token(constraint: str) -> str | None:
    if ":" not in constraint:
        return None
    _, rhs = constraint.split(":", 1)
    rhs = rhs.strip()
    if not rhs:
        return None
    # Use the first word on the RHS as a minimal, conservative gate.
    return rhs.split()[0]


def _extract_ids(worker_brief: str) -> list[str]:
    ids: list[str] = []
    for line in worker_brief.splitlines():
        line_s = line.strip()
        if line_s.lower().startswith("-") and "id:" in line_s.lower():
            # e.g. "- ID: ITEM_MAP_001"
            after = line_s.split(":", 1)[1].strip()
            if after:
                ids.append(after)
    return ids


def _iter_items(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, Iterable):
        return value
    return [value]
