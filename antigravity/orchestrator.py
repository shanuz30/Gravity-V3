"""
Antigravity orchestrator — unified pipeline.

Flow per query:
    1. Retrieve → set anchor (CRAG + LCV)
    2. Generate (GAN loop or direct)
    3. CRAG detect drift
    4. LCV compute damped correction
    5. If converged → return. Else → re-anchor and retry.

Clock-Time metacognition:
    Batch corrections over CLOCK_TIME_WINDOW steps.
    Equivalent to gamma*d damping on accumulated drift.
    Prevents reflexive-mode oscillation from token-level noise.

The joint attractor (d_Neural, d_Antigravity, d_Meta) → (0,0,0)
is the convergence target across all three levels.
"""

import time
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
import anthropic

from .core.types import AnchorState, CRAGSignal, GANSignal, LCVOutput
from .core.crag import CRAGDetector
from .core.lcv import LCVModule
from .core.gan import GANLoop


@dataclass
class AntigravityConfig:
    model: str = "claude-sonnet-4-20250514"

    # LCV (gamma-damping fix)
    gamma: float = 0.85
    eps_max: float = 0.5
    tau: float = 1.8
    alpha: float = 0.6
    convergence_threshold: float = 0.01

    # C-RAG thresholds
    js_threshold: float = 0.05
    cosine_threshold: float = 0.70

    # Clock-Time metacognition
    clock_time_window: int = 5       # batch N steps before applying correction
    max_correction_rounds: int = 3   # maximum re-anchor attempts

    # GAN
    gan_max_rounds: int = 3
    gan_confidence_floor: float = 0.40   # below this → do not trust correction

    # Tier 3 protection
    tier3_interrupt: bool = True


@dataclass
class PipelineState:
    """Full state snapshot per iteration — handoff artifact between agents."""
    iteration: int
    hypothesis: str
    anchor: Optional[AnchorState]
    gan: Optional[GANSignal]
    crag: Optional[CRAGSignal]
    lcv: Optional[LCVOutput]
    output: str = ""
    converged: bool = False
    alarm: str = "SAFE"


class Antigravity:
    """
    Unified Antigravity architecture.

    Inject `embed_fn` for your embedding model.
    Default: sentence-transformers all-MiniLM-L6-v2.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[AntigravityConfig] = None,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.config = config or AntigravityConfig()
        self.client = anthropic.Anthropic(api_key=api_key)

        self.lcv = LCVModule(
            eps_max=self.config.eps_max,
            tau=self.config.tau,
            alpha=self.config.alpha,
            gamma=self.config.gamma,
            convergence_threshold=self.config.convergence_threshold,
        )
        self.crag = CRAGDetector(
            js_threshold=self.config.js_threshold,
            cosine_threshold=self.config.cosine_threshold,
        )
        self.gan = GANLoop(
            client=self.client,
            model=self.config.model,
            max_rounds=self.config.gan_max_rounds,
        )

        self._embed_fn = embed_fn or self._default_embed
        self._embed_model = None  # lazy-loaded
        self._clock_buffer: list[np.ndarray] = []
        self._state_history: list[PipelineState] = []

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        hypothesis: str,
        retrieval_context: str,
        memory_context: str = "",
    ) -> PipelineState:
        """
        Full pipeline run.

        hypothesis       : the claim/question to reason about
        retrieval_context: RAG-retrieved content (becomes the anchor)
        memory_context   : memory.md + wiki.md contents
        """
        # 1. Set anchor from retrieval
        anchor = self._make_anchor(retrieval_context)
        anchor_emb = self._embed_fn(retrieval_context)
        anchor.embedding = anchor_emb

        self.lcv.set_anchor(anchor)
        self.crag.set_anchor(anchor, self._pseudo_token_dist(retrieval_context))

        state = PipelineState(
            iteration=0,
            hypothesis=hypothesis,
            anchor=anchor,
            gan=None, crag=None, lcv=None,
        )

        for i in range(self.config.max_correction_rounds):
            state.iteration = i

            # 2. GAN loop
            gan_signal = self.gan.run(hypothesis, memory_context)
            state.gan = gan_signal

            # Tier 3 protection
            if gan_signal.tier3_risk and self.config.tier3_interrupt:
                state.output = (
                    "TIER3_INTERRUPT: Recursive spiral detected. "
                    "Extract from current state: " + gan_signal.residual_truth
                )
                state.alarm = "TIER3"
                break

            # 3. Generate current output embedding
            current_text = gan_signal.residual_truth or gan_signal.core_claim
            current_emb = self._embed_fn(current_text)

            # Clock-Time: buffer or apply
            self._clock_buffer.append(current_emb)

            if len(self._clock_buffer) >= self.config.clock_time_window or i == 0:
                # Apply batch correction
                crag_signal = self.crag.detect(
                    np.mean(self._clock_buffer, axis=0),
                    self._pseudo_token_dist(current_text),
                )
                state.crag = crag_signal
                state.alarm = crag_signal.alarm

                # 4. LCV — damped correction
                if len(self._clock_buffer) >= self.config.clock_time_window:
                    lcv_out = self.lcv.batch_correct(
                        self._clock_buffer, gan_signal, crag_signal
                    )
                else:
                    lcv_out = self.lcv.compute(current_emb, gan_signal, crag_signal)

                state.lcv = lcv_out
                self._clock_buffer.clear()

                # 5. Convergence check
                if lcv_out.converged:
                    state.converged = True
                    state.output = self._format_output(gan_signal, lcv_out, crag_signal)
                    break

                # Not converged — check if GAN confidence too low to trust correction
                if gan_signal.confidence < self.config.gan_confidence_floor:
                    state.output = (
                        f"LOW_CONFIDENCE ({gan_signal.confidence:.0%}): "
                        "Trigger retrieval re-anchor before next correction."
                    )
                    break

                # Log oscillation risk
                osc = self.lcv.oscillation_risk()
                if osc > 0.7:
                    state.output = (
                        f"OSCILLATION_DETECTED (risk={osc:.2f}): "
                        "gamma may need reduction or retrieval re-anchor required."
                    )
                    break

        else:
            # Didn't converge in max rounds
            state.output = self._format_output(
                state.gan, state.lcv, state.crag, converged=False
            )

        self._state_history.append(state)
        return state

    # ── Utilities ─────────────────────────────────────────────────────────

    def _make_anchor(self, text: str) -> AnchorState:
        return AnchorState(
            embedding=np.zeros(384),  # filled after embed
            text=text,
            timestamp=time.time(),
            retrieval_id=str(uuid.uuid4())[:8],
        )

    def _pseudo_token_dist(self, text: str, vocab_size: int = 500) -> np.ndarray:
        """
        Approximate token distribution from text.
        Production: use actual logprobs from the LLM.
        """
        words = text.lower().split()
        dist = np.zeros(vocab_size)
        for w in words:
            idx = hash(w) % vocab_size
            dist[idx] += 1
        dist += 0.01  # Laplace smoothing
        return dist / dist.sum()

    def _format_output(
        self,
        gan: Optional[GANSignal],
        lcv: Optional[LCVOutput],
        crag: Optional[CRAGSignal],
        converged: bool = True,
    ) -> str:
        parts = []
        if gan:
            parts.append(f"CLAIM: {gan.core_claim}")
            parts.append(f"CONFIDENCE: {gan.confidence:.0%}")
            parts.append(f"RESIDUAL: {gan.residual_truth}")
        if lcv:
            parts.append(f"DRIFT: d={lcv.d:.4f} | damping={'active' if lcv.damping_active else 'inactive'}")
            parts.append(f"CONVERGED: {lcv.converged}")
        if crag:
            parts.append(f"CRAG: {crag.alarm} | JS={crag.js_divergence:.4f} | cos={crag.cosine_sim:.4f}")
        return "\n".join(parts)

    def _default_embed(self, text: str) -> np.ndarray:
        """Lazy-load sentence-transformers."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        v = self._embed_model.encode(text, normalize_embeddings=True)
        return np.array(v)

    @property
    def state_history(self) -> list[PipelineState]:
        return self._state_history.copy()

    def diagnostics(self) -> dict:
        return {
            "gamma": self.config.gamma,
            "current_d": self.lcv.current_d,
            "convergence_rate": self.lcv.convergence_rate(),
            "oscillation_risk": self.lcv.oscillation_risk(),
            "lcv_history_len": len(self.lcv.history),
        }
