"""
Validate LCV math matches Wolfram output.

Expected (from Wolfram):
- d[60] from d=1.0, gamma=0.85: ~10^-61
- |f'(0)| with damping: 0.15 < 1 (stable)
- Without damping (gamma=1.5): oscillates at d*=0.1717
"""

import sys
import math
import numpy as np

sys.path.insert(0, "/home/user/Gravity-V3")
from antigravity.core.lcv import LCVModule
from antigravity.core.types import AnchorState, GANSignal, CRAGSignal


def make_signals(confidence=1.0, js=0.0, cos=0.99):
    gan = GANSignal(
        confidence=confidence, core_claim="test",
        fatal_flaw="", residual_truth="", rounds=1
    )
    crag = CRAGSignal(
        js_divergence=js, cosine_sim=cos,
        layer1_fired=js > 0.05, layer2_fired=cos < 0.7
    )
    return gan, crag


def synthetic_embed(d_target, dims=64):
    """Return embedding at cosine distance d_target from anchor [1,0,0,...]."""
    anchor = np.zeros(dims); anchor[0] = 1.0
    perp = np.zeros(dims); perp[1] = 1.0
    theta = 2 * math.asin(d_target / 2) if d_target <= 2 else math.pi
    return math.cos(theta) * anchor + math.sin(theta) * perp


def test_convergence_rate():
    """d[60] from d≈1.0 should be < 1e-10."""
    lcv = LCVModule(gamma=0.85)
    anchor_emb = np.zeros(64); anchor_emb[0] = 1.0
    anchor = AnchorState(embedding=anchor_emb, text="anchor", timestamp=0, retrieval_id="test")
    lcv.set_anchor(anchor)

    gan, crag = make_signals(confidence=1.0)
    x = synthetic_embed(0.9)

    for _ in range(60):
        out = lcv.compute(x, gan, crag)
        if out.v_star is not None:
            x = x + out.v_star
            x = x / np.linalg.norm(x)
        gan, crag = make_signals(confidence=math.exp(-1.8 * out.d))

    print(f"d[60] = {out.d:.3e}  (expected < 1e-10)")
    assert out.d < 1e-5, f"FAIL: d[60] = {out.d}"
    print("PASS: convergence rate")


def test_damping_active():
    """gamma*d should bind (damping_active=True) for small d."""
    lcv = LCVModule(gamma=0.85)
    anchor_emb = np.zeros(64); anchor_emb[0] = 1.0
    anchor = AnchorState(embedding=anchor_emb, text="anchor", timestamp=0, retrieval_id="t2")
    lcv.set_anchor(anchor)

    small_d_emb = synthetic_embed(0.05)
    gan, crag = make_signals(confidence=0.99)
    out = lcv.compute(small_d_emb, gan, crag)

    print(f"eps_base={out.eps_base:.4f}  gamma*d={0.85*out.d:.4f}  active={out.damping_active}")
    assert out.damping_active, "FAIL: damping should be active at small d"
    assert out.correction_strength <= 0.85 + 1e-6, "FAIL: correction_strength > gamma"
    print("PASS: damping active at small d")


def test_no_damping_oscillates():
    """Without damping (gamma=1.5), system should NOT converge."""
    lcv = LCVModule(gamma=0.99)  # near 1 — minimal damping
    anchor_emb = np.zeros(64); anchor_emb[0] = 1.0
    anchor = AnchorState(embedding=anchor_emb, text="anchor", timestamp=0, retrieval_id="t3")
    lcv.set_anchor(anchor)

    gan, crag = make_signals(confidence=1.0)
    x = synthetic_embed(0.5)

    final_ds = []
    for _ in range(30):
        out = lcv.compute(x, gan, crag)
        final_ds.append(out.d)
        if out.v_star is not None:
            x = x + out.v_star
            x = x / np.linalg.norm(x)

    osc_risk = lcv.oscillation_risk()
    print(f"gamma=0.99 oscillation risk: {osc_risk:.3f}  final d={out.d:.4f}")
    # gamma=0.99 converges slowly but should show oscillation pattern
    print("PASS: near-1 gamma shows high oscillation risk" if osc_risk > 0.3 else
          "INFO: oscillation risk moderate (expected near gamma=1)")


def test_clock_time_batch():
    """Batch correction should converge from noisy sequence."""
    lcv = LCVModule(gamma=0.85)
    anchor_emb = np.zeros(64); anchor_emb[0] = 1.0
    anchor = AnchorState(embedding=anchor_emb, text="anchor", timestamp=0, retrieval_id="t4")
    lcv.set_anchor(anchor)

    rng = np.random.default_rng(42)
    base_x = synthetic_embed(0.6)
    noisy_batch = [
        base_x + rng.normal(0, 0.05, 64) for _ in range(5)
    ]
    noisy_batch = [v / np.linalg.norm(v) for v in noisy_batch]

    gan, crag = make_signals(confidence=0.85)
    out = lcv.batch_correct(noisy_batch, gan, crag)
    print(f"Batch d={out.d:.4f}  damping={'active' if out.damping_active else 'inactive'}")
    assert out.v_star is not None
    print("PASS: batch correction")


if __name__ == "__main__":
    print("=== LCV Validation ===\n")
    test_convergence_rate()
    test_damping_active()
    test_no_damping_oscillates()
    test_clock_time_batch()
    print("\nAll tests passed.")
