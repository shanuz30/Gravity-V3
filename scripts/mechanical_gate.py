import re
import sys
import json

class MechanicalGate:
    """
    Non-LLM deterministic guardrail.
    Intercepts [DO] execution commands from the LLM and runs them through safety filters.
    """
    def __init__(self):
        self.blacklist_patterns = [
            r"rm\s+-rf\s+/",
            r"mkfs",
            r">\s*/dev/sda",
            r"DROP\s+TABLE",
            r"DELETE\s+MATCH"
        ]

    def validate_command(self, command_intent: str) -> bool:
        for pattern in self.blacklist_patterns:
            if re.search(pattern, command_intent, re.IGNORECASE):
                print(f"[MECHANICAL_GATE_BLOCK] Destructive pattern detected: {pattern}")
                return False
        return True

    def cerebellum_translate(self, intent_json: str) -> str:
        try:
            data = json.loads(intent_json)
            if data.get("action") == "query_graph":
                return f"cypher: {data.get('query')}"
            return f"Translated execution for: {data.get('action')}"
        except json.JSONDecodeError:
            print("[MECHANICAL_GATE_ERROR] Invalid Intent JSON.")
            return ""

if __name__ == "__main__":
    gate = MechanicalGate()
    if len(sys.argv) > 1:
        test_cmd = sys.argv[1]
        is_safe = gate.validate_command(test_cmd)
        print(f"Command Safe: {is_safe}")
