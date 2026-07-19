"""LLM clients behind one thin ``LLMClient`` protocol (``model`` + ``complete``).

The orchestrator's *decisions* (recommendation, confidence, abstain) are derived
deterministically from retrieved evidence — auditable and independent of any model. An
``LLMClient`` only supplies the natural-language *narration* of each reasoning step. That
separation is deliberate: it keeps the scaffolding testable offline and makes the eventual
policy (Qwen) or teacher (Claude) a swappable narrator, not a hidden oracle.

- ``DeterministicLLM`` — offline, reproducible narrator. The default for CPU runs and tests.
- ``ClaudeLLM`` — the teacher slot. Deliberately unimplemented: the offline CPU pipeline does
  not need a teacher, so the slot raises a clear message rather than shipping an unreviewed
  API integration.
"""
from __future__ import annotations


class DeterministicLLM:
    """A fixed, dependency-free narrator. Echoes a compact, stable line so traces read
    naturally and reproducibly without a network call."""

    def __init__(self, model: str = "deterministic-v0") -> None:
        self.model = model

    def complete(self, prompt: str, **kw) -> str:
        # Stable: return the last non-empty line of the prompt (the orchestrator passes the
        # already-composed step summary as the final line). No randomness, no network.
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        return lines[-1] if lines else ""


class ClaudeLLM:
    """Teacher slot (claude-opus-4-8 / claude-sonnet-5). Intentionally unimplemented.

    The offline pipeline does not require a teacher, so the API integration is left unwired
    rather than shipped unreviewed. The interface it will satisfy is the same
    ``complete(prompt) -> str`` the orchestrator already uses.
    """

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        self.model = model

    def complete(self, prompt: str, **kw) -> str:
        raise NotImplementedError(
            "ClaudeLLM is a reserved teacher slot: wire the Anthropic client before using "
            "it. Use DeterministicLLM for offline runs."
        )
