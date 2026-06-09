"""
C-RAG drift detection — two-layer.

Layer 1: Jensen-Shannon divergence (token distribution vs anchor).
Layer 2: Cosine similarity to retrieval anchor embedding.

Upgrade from entropy delta (previous design):
- JS is bounded [0,1], symmetric, no division-by-zero
- Tripwire at JS > 0.05 (validated: clean separation in 200-sample Wolfram sim)
- Cosine tripwire at < 0.7

Both layers must fire for FULL_DRIFT alarm.
Either alone is a warning, not an alarm.
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from typing import Optional
from .types import CRAGSignal, AnchorState


class CRAGDetector:
    """
    Two-layer drift detector.

    Thresholds (from Wolfram phase-space analysis):
    - Layer 1: JS divergence > 0.05
    - Layer 2: cosine similarity < 0.70
    """

    def __init__(
        self,
        js_threshold: float = 0.05,
        cosine_threshold: float = 0.70,
        vocab_size: int = 1000,
    ):
        self.js_threshold = js_threshold
        self.cosine_threshold = cosine_threshold
        self.vocab_size = vocab_size

        self._anchor_dist: Optional[np.ndarray] = None
        self._anchor_embedding: Optional[np.ndarray] = None

    # ── Anchor setup ───────────────────────────────────────────────────────

    def set_anchor(self, anchor: AnchorState, token_dist: np.ndarray) -> None:
        """
        Set retrieval anchor.

        anchor     : AnchorState with normalized embedding
        token_dist : token probability distribution over vocab (Layer 1 reference)
        """
        self._anchor_embedding = anchor.embedding / np.linalg.norm(anchor.embedding)
        self._anchor_dist = self._normalize_dist(token_dist)

    # ── Detection ──────────────────────────────────────────────────────────

    def detect(
        self,
        current_embedding: np.ndarray,
        current_token_dist: Optional[np.ndarray] = None,
    ) -> CRAGSignal:
        """
        Run both detection layers.

        If current_token_dist is None, Layer 1 is skipped
        (cosine-only mode, safe fallback).
        """
        if self._anchor_embedding is None:
            raise RuntimeError("No anchor set. Call set_anchor() after retrieval.")

        # Layer 2: cosine similarity
        x = current_embedding / (np.linalg.norm(current_embedding) + 1e-10)
        cosine_sim = float(np.dot(self._anchor_embedding, x))

        # Layer 1: JS divergence
        if current_token_dist is not None and self._anchor_dist is not None:
            q = self._normalize_dist(current_token_dist)
            # scipy jensenshannon returns sqrt(JS) — square it for raw JS
            js_div = float(jensenshannon(self._anchor_dist, q) ** 2)
        else:
            js_div = 0.0

        return CRAGSignal(
            js_divergence=js_div,
            cosine_sim=cosine_sim,
            layer1_fired=js_div > self.js_threshold,
            layer2_fired=cosine_sim < self.cosine_threshold,
        )

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_dist(dist: np.ndarray) -> np.ndarray:
        """Ensure valid probability distribution."""
        dist = np.abs(dist)
        total = dist.sum()
        if total < 1e-10:
            return np.ones(len(dist)) / len(dist)
        return dist / total

    @staticmethod
    def token_dist_from_logprobs(logprobs: np.ndarray) -> np.ndarray:
        """Convert raw logprobs to probability distribution."""
        probs = np.exp(logprobs - np.max(logprobs))
        return probs / probs.sum()

    @staticmethod
    def embedding_from_text(text: str) -> np.ndarray:
        """
        Lightweight embedding fallback using sentence-transformers.
        For production: replace with your embedding model.
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model.encode(text, normalize_embeddings=True)
        except ImportError:
            # Fallback: hash-based pseudo-embedding (testing only)
            rng = np.random.default_rng(hash(text) % (2**32))
            v = rng.normal(0, 1, 384)
            return v / np.linalg.norm(v)
