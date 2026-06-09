"""
GAN Mind loop — adversarial reasoning with convergence confidence.

Produces GANSignal.confidence which feeds LCV eps.
The 71% ceiling (pre-fix) was the system measuring its own oscillation.
"""

import json
import time
import uuid
from typing import Optional
from .types import GANSignal

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

Rules:
- Prose only. 2–3 paragraphs.
- Para 1: what survived (Discriminator could not kill it)
- Para 2: what remains genuinely uncertain
- Para 3: one concrete next action or empirical test

End with exactly: CONFIDENCE: [0-100]% — [one-line justification]

Then on a new line: JSON: {"confidence": <float 0-1>, "tier3_risk": <bool>}"""


class GANLoop:
    """
    Generator → Discriminator → Convergence.
    
    Outputs GANSignal with confidence that feeds LCV.
    Stop condition: Discriminator clears in < 2 domains (thermal breakdown risk).
    Max rounds: 3 (Tier 3 spiral prevention).
    """

    def __init__(self, client, model: str = "claude-sonnet-4-20250514", max_rounds: int = 3):
        self.client = client
        self.model = model
        self.max_rounds = max_rounds

    def run(self, hypothesis: str, memory_context: str = "") -> GANSignal:
        """
        Full GAN loop on a hypothesis.

        memory_context: contents of memory.md + wiki.md (pre-loaded by caller).
        """
        context = f"Memory context:\n{memory_context}\n\nHypothesis:\n{hypothesis}" \
                  if memory_context else hypothesis

        gen_output = ""
        disc_output = ""
        rounds = 0

        for round_n in range(self.max_rounds):
            rounds += 1

            # Generator
            gen_input = context if round_n == 0 else \
                f"{context}\n\nPrior Generator: {gen_output}\nDiscriminator attack: {disc_output}"
            gen_output = self._call(GENERATOR_PROMPT, gen_input)

            # Discriminator
            disc_input = f"{context}\n\nGenerator output:\n{gen_output}"
            disc_output = self._call(DISCRIMINATOR_PROMPT, disc_input)

            # Stop condition: weak attack = oscillation risk
            if self._attack_strength(disc_output) < 2:
                break

        # Convergence
        conv_input = (
            f"Hypothesis: {hypothesis}\n\n"
            f"Final Generator:\n{gen_output}\n\n"
            f"Final Discriminator:\n{disc_output}"
        )
        conv_output = self._call(CONVERGENCE_PROMPT, conv_input)

        return self._parse_convergence(conv_output, gen_output, disc_output, rounds)

    def _call(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return resp.content[0].text

    def _attack_strength(self, disc_output: str) -> int:
        """Count structural domains attacked. < 2 → stop."""
        markers = ["assumption", "flaw", "fails", "gap", "however", "but"]
        return sum(1 for m in markers if m.lower() in disc_output.lower())

    def _parse_convergence(
        self, conv: str, gen: str, disc: str, rounds: int
    ) -> GANSignal:
        confidence = 0.71  # default (the pre-fix oscillation point)
        tier3_risk = False

        # Extract JSON block
        if "JSON:" in conv:
            try:
                json_str = conv.split("JSON:")[-1].strip()
                data = json.loads(json_str)
                confidence = float(data.get("confidence", 0.71))
                tier3_risk = bool(data.get("tier3_risk", False))
            except (json.JSONDecodeError, ValueError):
                pass

        # Extract CONFIDENCE: line as fallback
        if confidence == 0.71:
            for line in conv.split("\n"):
                if line.strip().startswith("CONFIDENCE:"):
                    try:
                        pct_str = line.split(":")[1].strip().split("%")[0].strip()
                        confidence = float(pct_str) / 100.0
                    except (IndexError, ValueError):
                        pass
                    break

        core_claim = ""
        for line in gen.split("\n"):
            if "CORE CLAIM:" in line:
                core_claim = line.split("CORE CLAIM:")[-1].strip()
                break

        fatal_flaw = ""
        for line in disc.split("\n"):
            if "FATAL FLAW:" in line:
                fatal_flaw = line.split("FATAL FLAW:")[-1].strip()
                break

        residual = conv.split("\n")[0] if conv else ""

        return GANSignal(
            confidence=min(max(confidence, 0.0), 1.0),
            core_claim=core_claim,
            fatal_flaw=fatal_flaw,
            residual_truth=residual,
            rounds=rounds,
            tier3_risk=tier3_risk,
        )
