# Antigravity Architecture

## What this is

A convergence-stable agentic reasoning system. Three coupled components:

- **GAN Mind** — Generator → Discriminator → Convergence adversarial reasoning loop
- **C-RAG** — two-layer drift detection (JS divergence + cosine similarity)
- **LCV** — Learned Correction Vector with gamma-damping fix

The critical insight from formal analysis: without the gamma-damping fix, the system
has an unstable fixed point at d*≈0.17 (GAN confidence ≈71%) and oscillates.
With it: globally stable, converges to d=0 in ~60 iterations from anywhere.

The fix is one constraint: `v* = min(eps(g, CRAG), 0.85 * ||x-r||) * normalize(r-x)`

## Status

**Built and tested:**
- `antigravity/core/types.py` — dataclasses for every agent boundary
- `antigravity/core/lcv.py` — LCV with gamma-damping (validate.py passes)
- `antigravity/core/crag.py` — C-RAG drift detector (JS + cosine)
- `antigravity/core/gan.py` — GAN loop structure (API integration untested)
- `antigravity/orchestrator.py` — main pipeline with Clock-Time metacognition

**Not yet built:**
- memory.md / wiki.md file persistence (stubs exist, not wired)
- Real logprob → token distribution for C-RAG Layer 1
- MySQL MCP as shared blackboard (swarm operation)
- Wolfram verification step in GAN Convergence

## Key parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| gamma | 0.85 | Wolfram sweep: optimal convergence rate |
| JS threshold | 0.05 | Phase-space analysis: clean separation |
| cosine threshold | 0.70 | Phase-space analysis: large gap between regimes |
| clock_time_window | 5 | Prevents reflexive-mode oscillation from noise |
| max_rounds | 3 | Tier-3 spiral prevention |
| gan_confidence_floor | 0.40 | Below this: re-anchor, don't correct |

## Architecture diagram

```
User query
    │
    ▼
[Retrieval] → set_anchor(r) → CRAG.set_anchor + LCV.set_anchor
    │
    ▼
[GAN Loop] → Generator → Discriminator → Convergence → GANSignal{confidence, claim}
    │                                         │
    │                              write to memory.md + wiki.md
    ▼
[CRAG detect] → CRAGSignal{js_divergence, cosine_sim, alarm}
    │
    ▼
[LCV compute] → v* = min(eps(g,CRAG), 0.85*d) * normalize(r-x)
    │
    ├── converged? → return output
    └── not converged? → re-anchor if confidence < 0.40, else iterate
```

## Swarm operation (not yet built)

For multi-agent swarming, memory.md and wiki.md move from local files to MySQL.
Joint Lyapunov stability proven: V = Σd_i² decreases monotonically.
Swarm benefit is collective knowledge (shared wiki), not faster individual convergence.

```
Subclass Brain 1  ─┐
Subclass Brain 2  ──→ MySQL shared blackboard → Superclass Brain distillation
Subclass Brain N  ─┘
```
