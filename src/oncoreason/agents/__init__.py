"""Agent scaffolding: trace schema + tools + multi-agent orchestrator."""
from .clients import ClaudeLLM, DeterministicLLM
from .orchestrator import SPECIALISTS, LLMClient, Orchestrator
from .tools import guideline_lookup, variant_lookup
from .trace import Citation, ReasoningStep, ToolCall, Trace

__all__ = [
    "Trace",
    "ReasoningStep",
    "ToolCall",
    "Citation",
    "Orchestrator",
    "LLMClient",
    "SPECIALISTS",
    "DeterministicLLM",
    "ClaudeLLM",
    "variant_lookup",
    "guideline_lookup",
]
