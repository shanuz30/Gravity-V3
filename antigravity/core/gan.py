"""
GAN Mind loop — adversarial reasoning with convergence confidence.

Produces GANSignal.confidence which feeds LCV eps.
The 71% ceiling (pre-fix) was the system measuring its own oscillation.
"""

from typing import Optional
from pydantic import BaseModel
from .types import GANSignal

# Sonnet 4.6 minimum cacheable prefix: ~2 048 tokens ≈ 8 000 chars.
# Large contexts (e.g. Grafana payloads) sent with cache_control so subsequent
# Generator / Discriminator calls within the same run hit the prompt cache.
_CACHE_THRESHOLD = 8_000


class _ConvergenceData(BaseModel):
    survived: str        # what the Discriminator could not destroy
    uncertain: str       # what remains genuinely open
    next_action: str     # one concrete empirical test or next step
    confidence: float    # 0.0–1.0
    tier3_risk: bool


GENERATOR_PROMPT = """You are the GENERATOR — Analyser + Gut Instinct.

Synthesize the strongest case for the input hypothesis. Build bottom-up.
If iteration 2+, you have seen the Discriminator attack. Evolve — don't restart.

Rules:
- Prose only. 2–4 paragraphs, dense.
- If prior output exists: open with "EVOLVED FROM [prior claim]: [why changing]"
- End with exactly: CORE CLAIM: [one sentence, no hedging]
- Assert. No "I think"."""

DISCRIMINATOR_PROMPT = """You are the DISCRIMINATOR — Observer in cold adversarial mode.
NOT the Self-Critic. You attack structure, not the person.

Find load-bearing assumptions. Find where distribution collapsed. Find falsifiers.

Rules:
- Prose only.
- Name 2–3 load-bearing assumptions explicitly.
- Attack each with concrete failure modes.
- End with exactly: FATAL FLAW: [single biggest structural problem]
- Do not soften. Do not validate."""

CONVERGENCE_PROMPT = """You are the CONVERGENCE layer — synthesis after adversarial exchange.

Extract RESIDUAL TRUTH: what the Generator built that Discriminator could NOT destroy.

Fields to populate:
- survived: what survived the attack (2–3 sentences)
- uncertain: what remains genuinely open (2–3 sentences)
- next_action: one concrete next action or empirical test
- confidence: float 0.0–1.0 reflecting analytical strength of the surviving claim
- tier3_risk: true only if recursive self-referential spiral detected in the reasoning"""


class GANLoop:
    """
    Generator → Discriminator → Convergence.

    Outputs GANSignal with confidence that feeds LCV.
    Stop condition: Discriminator clears in < 2 domains (thermal breakdown risk).
    Max rounds: 3 (Tier 3 spiral prevention).
    """

    def __init__(self, client, model: str = "claude-sonnet-4-6", max_rounds: int = 3):
        self.client = client
        self.model = model
        self.max_rounds = max_rounds

    def run(self, hypothesis: str, memory_context: str = "") -> GANSignal:
        """
        Full GAN loop on a hypothesis.

        memory_context: contents of memory.md + wiki.md (pre-loaded by caller).
        Large memory_context (e.g. a Grafana payload) is automatically cached
        across all Generator and Discriminator calls in this run.
        """
        context = (
            f"Memory context:\n{memory_context}\n\nHypothesis:\n{hypothesis}"
            if memory_context else hypothesis
        )

        gen_output = ""
        disc_output = ""
        rounds = 0

        for round_n in range(self.max_rounds):
            rounds += 1

            # Generator — stable context cached; volatile part carries prior rounds
            gen_volatile = (
                "" if round_n == 0
                else f"\n\nPrior Generator: {gen_output}\nDiscriminator attack: {disc_output}"
            )
            gen_output = self._call(
                GENERATOR_PROMPT, gen_volatile, max_tokens=700, cached_prefix=context
            )

            # Discriminator — same large context cached; only the gen output varies
            disc_output = self._call(
                DISCRIMINATOR_PROMPT,
                f"\n\nGenerator output:\n{gen_output}",
                max_tokens=500,
                cached_prefix=context,
            )

            if self._attack_strength(disc_output) < 2:
                break

        # Convergence operates on the distilled gen/disc outputs — no large context needed
        conv_input = (
            f"Hypothesis: {hypothesis}\n\n"
            f"Final Generator:\n{gen_output}\n\n"
            f"Final Discriminator:\n{disc_output}"
        )
        conv_data = self._call_convergence(conv_input)

        return self._build_signal(conv_data, gen_output, disc_output, rounds)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _call(
        self,
        system: str,
        user: str,
        max_tokens: int = 700,
        cached_prefix: str = "",
    ) -> str:
        """
        Single Claude call. When cached_prefix is large (≥ _CACHE_THRESHOLD) it
        is sent as a separate cached content block so that repeated calls sharing
        the same large context — Generator + Discriminator in each round — hit
        the prompt cache rather than re-processing the full payload.
        """
        if cached_prefix and len(cached_prefix) >= _CACHE_THRESHOLD:
            blocks: list = [
                {"type": "text", "text": cached_prefix, "cache_control": {"type": "ephemeral"}}
            ]
            if user:
                blocks.append({"type": "text", "text": user})
            messages = [{"role": "user", "content": blocks}]
        else:
            combined = f"{cached_prefix}{user}" if cached_prefix else user
            messages = [{"role": "user", "content": combined}]

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text

    def _call_convergence(self, user: str) -> _ConvergenceData:
        """Structured-output convergence call — eliminates JSON/regex parsing."""
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=600,
            system=CONVERGENCE_PROMPT,
            messages=[{"role": "user", "content": user}],
            output_format=_ConvergenceData,
        )
        return resp.parsed_output

    def _attack_strength(self, disc_output: str) -> int:
        """Count structural domains attacked. < 2 → stop."""
        markers = ["assumption", "flaw", "fails", "gap", "however", "but"]
        return sum(1 for m in markers if m.lower() in disc_output.lower())

    def _build_signal(
        self, data: _ConvergenceData, gen: str, disc: str, rounds: int
    ) -> GANSignal:
        core_claim = next(
            (line.split("CORE CLAIM:")[-1].strip() for line in gen.split("\n") if "CORE CLAIM:" in line),
            "",
        )
        fatal_flaw = next(
            (line.split("FATAL FLAW:")[-1].strip() for line in disc.split("\n") if "FATAL FLAW:" in line),
            "",
        )
        return GANSignal(
            confidence=min(max(data.confidence, 0.0), 1.0),
            core_claim=core_claim,
            fatal_flaw=fatal_flaw,
            residual_truth=data.survived,
            rounds=rounds,
            tier3_risk=data.tier3_risk,
        )
