"""
DiffServe: Online Serving for Diffusion LLMs with CW-SRPT Scheduling.

A dLLM-native serving system that exploits per-token confidence information
for Confidence-Weighted SRPT scheduling, achieving up to 84.7% latency
reduction over FCFS with zero output quality loss.
"""

from .config import DiffServeConfig
from .request import DiffuseRequest
from .scheduler import SchedulingPolicy, pick_batch

__all__ = [
    "DiffServeConfig",
    "DiffuseRequest",
    "SchedulingPolicy",
    "pick_batch",
]
