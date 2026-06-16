"""
DiffuseRequest: Per-request state for iteration-level scheduling.

Each request tracks its own token sequence, block progress, and confidence
metrics for CW-SRPT scheduling. Requests are preemptible at iteration
boundaries — the scheduler can pick a different set of requests each iteration.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .config import DiffServeConfig


@dataclass
class DiffuseRequest:
    """A single dLLM inference request with iteration-level scheduling state.

    Attributes:
        id: Unique request identifier.
        prompt_ids: Tokenized prompt tensor on device.
        gen_length: Number of tokens to generate.
        threshold: Confidence threshold for unmasking.
        config: Reference to global config (for mask_id, eos_id, block_length).
    """

    id: int
    prompt_ids: torch.Tensor
    gen_length: int
    threshold: float
    config: DiffServeConfig

    # ─── Sequence state ───────────────────────────────────────────
    x: Optional[torch.Tensor] = field(default=None, repr=False)
    prompt_len: int = 0
    total_len: int = 0
    cur_block: int = 0
    n_blocks: int = 0

    # ─── Timing ───────────────────────────────────────────────────
    arrival_time: float = 0.0
    first_token_time: float = -1.0
    finish_time: float = -1.0
    total_fwds: int = 0

    # ─── CW-SRPT scheduling state ────────────────────────────────
    last_confidence_sum: float = 0.0    # sum of confidence at masked positions
    last_n_unmasked: int = 0            # tokens unmasked in last iteration
    convergence_rate: float = 0.0       # EMA of tokens_unmasked / iteration
    iters_since_progress: int = 0       # starvation detector (for APPC)

    # ─── KV cache state (when use_kv_cache=True) ────────────────
    prefilled: bool = False  # whether KV cache has been populated
    kv_seq_len: int = 0      # current cached sequence length (prefix only)

    # ─── Output ───────────────────────────────────────────────────
    output_text: str = ""

    def init(self):
        """Initialize the token sequence with prompt + masked generation slots."""
        device = self.prompt_ids.device
        mask_id = self.config.mask_id
        block_length = self.config.block_length

        self.prompt_len = self.prompt_ids.shape[0]
        self.n_blocks = (self.gen_length + block_length - 1) // block_length
        self.total_len = self.prompt_len + self.n_blocks * block_length

        # Create full sequence: [prompt tokens | MASK tokens...]
        self.x = torch.full(
            (self.total_len,), mask_id, dtype=torch.long, device=device)
        self.x[:self.prompt_len] = self.prompt_ids

        self.arrival_time = time.perf_counter() * 1000  # ms

    # ─── Block boundaries ─────────────────────────────────────────

    @property
    def block_start(self) -> int:
        """Start index of current denoising block."""
        return self.prompt_len + self.cur_block * self.config.block_length

    @property
    def block_end(self) -> int:
        """End index of current denoising block."""
        return min(
            self.block_start + self.config.block_length, self.total_len)

    # ─── Status ───────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        """Whether this request has completed all blocks."""
        return self.cur_block >= self.n_blocks

    @property
    def n_masked(self) -> int:
        """Number of remaining masked tokens in current block."""
        if self.x is None or self.done:
            return 0
        return int((self.x[self.block_start:self.block_end] == self.config.mask_id).sum().item())

    # ─── Scheduling priorities ────────────────────────────────────

    @property
    def remaining_naive(self) -> int:
        """SRPT priority: total remaining masked tokens (current + future blocks)."""
        return self.n_masked + (self.n_blocks - self.cur_block - 1) * self.config.block_length

    @property
    def remaining_confidence(self) -> float:
        """CW-SRPT priority: confidence-weighted remaining work.

        Key insight: positions with high confidence are about to be unmasked,
        so their "effective remaining work" is lower. This is dLLM-specific
        information that AR LLMs don't have.

        Formula: n_masked * (1 - avg_confidence) + blocks_left * block_length
        """
        blocks_left = self.n_blocks - self.cur_block - 1
        base = blocks_left * self.config.block_length
        nm = self.n_masked
        if self.last_confidence_sum > 0 and nm > 0:
            avg_conf = self.last_confidence_sum / nm
            return nm * (1.0 - avg_conf) + base
        return nm + base

    @property
    def progress_priority(self) -> float:
        """APPC priority: inverse convergence rate (fast convergers run first).

        Higher convergence_rate → lower priority number → runs first.
        """
        if self.convergence_rate > 0:
            return self.n_masked / max(self.convergence_rate, 0.1)
        return float(self.remaining_naive)

    # ─── Lifecycle ────────────────────────────────────────────────

    def advance_block(self):
        """Advance to the next block after current block is fully decoded."""
        self.cur_block += 1

    def mark_early_stop(self):
        """Mark request as done due to EOS token in current block."""
        eos_id = self.config.eos_id
        self.x[self.block_end:] = eos_id
        self.cur_block = self.n_blocks

    def decode_output(self, tokenizer) -> str:
        """Decode the generated tokens into text."""
        gen = self.x[self.prompt_len:]
        eos_id = self.config.eos_id
        eos_pos = (gen == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            gen = gen[:eos_pos[0]]
        self.output_text = tokenizer.decode(gen, skip_special_tokens=True)
        return self.output_text
