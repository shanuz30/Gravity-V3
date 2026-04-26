import json
import sys

class QueryArbitrator:
    """
    [DUAL-BRAIN ROUTER]
    Routes queries between System 1 (Vector/Intuition) and System 2 (Graph/Structure).
    Ensures deterministic grounding by bridging the 'Domain Gap' between semantics and topology.
    """
    def __init__(self):
        # System 2: Structural/Topological/Deterministic (Memgraph)
        self.system_2_triggers = [
            "depends", "imports", "calls", "structure", "topology", 
            "connected", "blocked", "path", "hierarchy", "graph",
            "relationship", "upstream", "downstream", "impact"
        ]
        
        # System 1: Conceptual/Semantic/Probabilistic (Qdrant)
        self.system_1_triggers = [
            "why", "explain", "how", "meaning", "history", 
            "concept", "general", "summary", "decision", "rationale",
            "background", "context"
        ]

    def arbitrate(self, query: str) -> dict:
        query_lower = query.lower()
        s1_hits = sum(1 for k in self.system_1_triggers if k in query_lower)
        s2_hits = sum(1 for k in self.system_2_triggers if k in query_lower)
        
        if s2_hits > s1_hits:
            decision = {
                "target": "MEMGRAPH",
                "paradigm": "System 2 (Deterministic/Structural)",
                "confidence": min(1.0, s2_hits / 3.0),
                "rationale": f"Detected {s2_hits} structural indicators."
            }
        elif s1_hits > 0:
            decision = {
                "target": "QDRANT",
                "paradigm": "System 1 (Semantic/Conceptual)",
                "confidence": min(1.0, s1_hits / 3.0),
                "rationale": f"Detected {s1_hits} conceptual indicators."
            }
        else:
            decision = {
                "target": "HYBRID",
                "paradigm": "Low Confidence / Default",
                "confidence": 0.5,
                "rationale": "No strong triggers detected. Running parallel search."
            }
        return decision

if __name__ == "__main__":
    arbitrator = QueryArbitrator()
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        result = arbitrator.arbitrate(user_query)
        print(json.dumps(result, indent=2))
