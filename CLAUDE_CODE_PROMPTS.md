# Claude Code Prompts — Antigravity Implementation

Each prompt is a standalone task. Run them in order.
All files are in `antigravity/`. Read `CLAUDE.md` first.

---

## Prompt 1: Wire memory.md and wiki.md to disk

**Task:** The GAN loop in `antigravity/core/gan.py` produces GANSignal but doesn't
persist anything. Wire memory.md and wiki.md to actual files.

**Files in scope:**
- `antigravity/core/gan.py` (modify)
- `antigravity/state/` (create directory + files)

**What to build:**
1. On each GANLoop.run() call, append to `antigravity/state/memory.md`:
   ```
   # GAN Loop: {hypothesis[:50]} | {timestamp}
   ## Meta
   - Hypothesis ID: {uuid}
   - Confidence: {signal.confidence:.0%}
   - Rounds: {signal.rounds}
   ## Core Claim
   {signal.core_claim}
   ## Fatal Flaw
   {signal.fatal_flaw}
   ## Residual Truth
   {signal.residual_truth}
   ---
   ```

2. After Convergence, if confidence > 0.85, append new pattern to
   `antigravity/state/wiki.md` (only unique claims, check for duplicates first).

3. At start of each run(), inject last 3 memory.md entries + full wiki.md
   into the `memory_context` if none was passed by caller.

**Success criteria:**
- `state/memory.md` grows after each `gan.run()` call
- `state/wiki.md` only updates when confidence > 0.85
- Re-running same hypothesis returns richer context on second call

**Do not touch:** `core/types.py`, `core/lcv.py`, `core/crag.py`

---

## Prompt 2: Test GAN loop against real Anthropic API

**Task:** `antigravity/core/gan.py` is structurally sound but untested against
real API responses. Build a test that calls the real API with a known hypothesis.

**Files in scope:**
- `antigravity/test_gan_real.py` (create)

**What to build:**
```python
# test_gan_real.py
# Requires ANTHROPIC_API_KEY in environment
# Runs ONE real GAN loop and validates output structure

import anthropic
from antigravity.core.gan import GANLoop

client = anthropic.Anthropic()
gan = GANLoop(client, max_rounds=1)

hypothesis = "The unstable fixed point in RAG correction systems is caused by
unconstrained correction magnitude, not retrieval quality."

signal = gan.run(hypothesis, memory_context="")

# Assertions
assert 0.0 <= signal.confidence <= 1.0
assert len(signal.core_claim) > 10
assert len(signal.fatal_flaw) > 10
assert signal.rounds >= 1
assert not signal.tier3_risk  # should not spiral on first run

print(f"Confidence: {signal.confidence:.0%}")
print(f"Claim: {signal.core_claim}")
print(f"Flaw: {signal.fatal_flaw}")
print(f"Rounds: {signal.rounds}")
```

**Success criteria:** Script runs without error, all assertions pass.

**Note:** Check that CONVERGENCE_PROMPT JSON parsing works against real output.
The current parser has a fallback to 0.71 — that fallback should NOT trigger
on real API responses. If it does, fix the JSON extraction in `_parse_convergence`.

---

## Prompt 3: Replace pseudo token distribution with real logprobs

**Task:** `antigravity/orchestrator.py` uses `_pseudo_token_dist()` which hashes
words — not real token probabilities. C-RAG Layer 1 (JS divergence) is currently
approximate. Replace with real logprobs from the API.

**Files in scope:**
- `antigravity/orchestrator.py` (modify)
- `antigravity/core/crag.py` (modify — add logprob_to_dist utility)

**What to build:**
1. In `orchestrator.py`, when generating text via the GAN loop, capture logprobs:
   ```python
   # Use Anthropic API with top_k logprobs
   resp = client.messages.create(
       model=model,
       max_tokens=100,
       messages=[...],
       # Note: check current API docs for logprob support
   )
   ```

2. If Anthropic API doesn't expose logprobs directly, use output token
   length distribution as a proxy:
   - Count token frequencies in the output text
   - Use this as the empirical distribution for JS divergence

3. Update `CRAGDetector.detect()` to accept `Optional[np.ndarray]` token_dist.
   When None: skip Layer 1, cosine-only mode (already implemented as fallback).

**Success criteria:**
- Layer 1 fires on semantically drifted output (test with clearly off-topic input)
- Layer 1 does NOT fire on anchored output (test with same-domain input)
- `crag_signal.layer1_fired` behaves predictably

**Check first:** Whether current Anthropic API supports logprob output.
If not: document this clearly and keep the frequency-count proxy.

---

## Prompt 4: MySQL shared blackboard for swarm operation

**Task:** Move memory.md and wiki.md from local files to MySQL via the existing
MCP server. This enables multi-instance swarm operation.

**Files in scope:**
- `antigravity/state/blackboard.py` (create)
- `antigravity/core/gan.py` (modify — use blackboard instead of files)
- `antigravity/orchestrator.py` (modify — pass blackboard to gan)

**What to build:**

```python
# antigravity/state/blackboard.py

class Blackboard:
    """
    Unified interface for memory.md + wiki.md.
    Defaults to local files. Set shared=True for MySQL swarm mode.
    """

    def __init__(self, shared: bool = False, mysql_config: dict = None):
        self.shared = shared
        self.mysql_config = mysql_config
        self._local_memory_path = "antigravity/state/memory.md"
        self._local_wiki_path = "antigravity/state/wiki.md"

    def append_memory(self, entry: str, instance_id: str = "default") -> None:
        if self.shared:
            self._mysql_write("memory", instance_id, entry)
        else:
            self._file_append(self._local_memory_path, entry)

    def get_memory(self, last_n: int = 3, instance_id: str = None) -> str:
        """Returns last_n entries. If shared + instance_id=None: all instances."""
        if self.shared:
            return self._mysql_read("memory", last_n, instance_id)
        return self._file_read_last_n(self._local_memory_path, last_n)

    def update_wiki(self, pattern: str, confidence: float) -> None:
        """Only writes if confidence > 0.85 and pattern is novel."""
        if confidence < 0.85:
            return
        existing = self.get_wiki()
        if pattern[:50] in existing:
            return  # dedup
        entry = f"\n## Pattern [{confidence:.0%}]\n{pattern}\n"
        if self.shared:
            self._mysql_write("wiki", "global", entry)
        else:
            self._file_append(self._local_wiki_path, entry)

    def get_wiki(self) -> str:
        if self.shared:
            return self._mysql_read("wiki", None, "global")
        return self._file_read(self._local_wiki_path)

    def _mysql_write(self, table, key, value): ...  # implement with mysql-connector
    def _mysql_read(self, table, limit, key): ...
    def _file_append(self, path, content): ...
    def _file_read_last_n(self, path, n): ...
    def _file_read(self, path): ...
```

**MySQL schema:**
```sql
CREATE TABLE antigravity_memory (
    id INT AUTO_INCREMENT PRIMARY KEY,
    instance_id VARCHAR(64),
    content TEXT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE antigravity_wiki (
    id INT AUTO_INCREMENT PRIMARY KEY,
    instance_id VARCHAR(64) DEFAULT 'global',
    pattern TEXT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Success criteria:**
- `Blackboard(shared=False)` works identically to current file-based state
- `Blackboard(shared=True)` writes to MySQL and reads across instances
- Two instances of orchestrator with same MySQL config see each other's wiki

---

## Prompt 5: End-to-end validation suite

**Task:** `validate.py` only tests LCV math. Build a full pipeline test.

**Files in scope:**
- `antigravity/validate_full.py` (create)

**What to build:**
```python
# validate_full.py
# Tests the full pipeline without real API calls
# Uses mock GAN signals at known confidence values

from antigravity.core.lcv import LCVModule
from antigravity.core.crag import CRAGDetector
from antigravity.core.types import AnchorState, GANSignal, CRAGSignal
import numpy as np

def mock_gan(confidence: float) -> GANSignal:
    return GANSignal(
        confidence=confidence, core_claim="test claim",
        fatal_flaw="test flaw", residual_truth="test truth", rounds=1
    )

# Test 1: Pipeline at 71% confidence (pre-fix oscillation point)
# Expected: damping_active=True, system does NOT oscillate
def test_71_pct_no_oscillation(): ...

# Test 2: Pipeline at 40% confidence (below floor)
# Expected: converged=False, no correction applied
def test_below_floor(): ...

# Test 3: CRAG alarm triggers correctly
# Expected: FULL_DRIFT alarm when JS > 0.05 AND cosine < 0.70
def test_crag_alarm(): ...

# Test 4: Clock-Time batch (5 noisy embeddings → one correction)
# Expected: batch_correct() returns stable d, not noisy d
def test_clock_time_batch(): ...

# Test 5: Oscillation detection
# Expected: oscillation_risk() > 0.5 after alternating d values
def test_oscillation_detection(): ...
```

**Success criteria:** All 5 tests pass. No real API calls required.
