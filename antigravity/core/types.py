"""
Shared dataclasses for all Antigravity agent boundaries.
Every cross-module signal passes through one of these types.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AnchorState:
    """Retrieval anchor — the ground-truth reference point."""
    embedding: np.ndarray
    text: str
    timestamp: float
    retrieval_id: str


@dataclass
class GANSignal:
    """Output of one full GAN loop (Generator → Discriminator → Convergence)."""
    confidence: float       # [0, 1] — feeds LCV eps
    core_claim: str         # Generator's CORE CLAIM line
    fatal_flaw: str         # Discriminator's FATAL FLAW line
    residual_truth: str     # Convergence survivor
    rounds: int             # How many adversarial rounds ran
    tier3_risk: bool = False


@dataclass
class CRAGSignal:
    """Two-layer drift detection result."""
    js_divergence: float    # Layer 1 — JS divergence from anchor dist
    cosine_sim: float       # Layer 2 — cosine similarity to anchor embedding
    layer1_fired: bool      # JS > threshold
    layer2_fired: bool      # cosine < threshold

    @property
    def alarm(self) -> str:
        """
        FULL_DRIFT       — both layers fired (hard alarm, re-anchor required)
        ENTROPY_TRIPWIRE — JS divergence only (distribution drift)
        SEMANTIC_DRIFT   — cosine only (embedding drift)
        SAFE             — no drift detected
        """
        if self.layer1_fired and self.layer2_fired:
            return "FULL_DRIFT"
        if self.layer1_fired:
            return "ENTROPY_TRIPWIRE"
        if self.layer2_fired:
            return "SEMANTIC_DRIFT"
        return "SAFE"

    @property
    def drift_score(self) -> float:
        """Composite drift magnitude in [0, 1]. Both layers equally weighted."""
        return (self.js_divergence / 0.3 + (1 - self.cosine_sim)) / 2


@dataclass
class LCVOutput:
    """Result of one LCV compute() call — full diagnostic snapshot."""
    d: float                    # Distance to anchor
    eps_base: float             # Unconstrained correction magnitude
    eps_actual: float           # After gamma*d damping: min(eps_base, gamma*d)
    damping_active: bool        # True when gamma*d bound is binding
    correction_strength: float  # eps_actual / d — must be ≤ gamma
    converged: bool             # d < convergence_threshold
    v_star: np.ndarray          # Correction vector in embedding space
