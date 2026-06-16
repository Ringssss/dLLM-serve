"""DiffServe configuration."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DiffServeConfig:
    """Configuration for DiffServe online serving system.

    Attributes:
        model_path: Path to the dLLM model weights.
        mask_id: Token ID used for masked positions.
        eos_id: Token ID for end-of-sequence.
        block_length: Number of tokens per denoising block.
        max_batch_size: Maximum number of requests per batch iteration.
        max_seq_length: Maximum total sequence length (prompt + generation).
        threshold: Confidence threshold for unmasking tokens.
        policy: Scheduling policy name.
        supported_batch_sizes: Batch sizes to capture CUDA graphs for.
        prefill_lengths: Prefill sequence lengths to capture CUDA graphs for.
        host: Server bind address.
        port: Server listen port.
        device: CUDA device string (e.g., "cuda:0").
        enable_cuda_graph: Whether to use CUDA graph replay.
        enable_torch_compile: Whether to use torch.compile for model forward.
    """

    # Model
    model_path: str = "/mnt/models/LLaDA2.0-mini"
    mask_id: int = 156895
    eos_id: int = 156892
    block_length: int = 32

    # Serving
    max_batch_size: int = 8
    max_seq_length: int = 512
    threshold: float = 0.9
    policy: str = "cw-srpt"  # fcfs | srpt | cw-srpt | bab-srpt | appc

    # CUDA Graph
    supported_batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    prefill_lengths: List[int] = field(default_factory=lambda: [32, 64, 96, 128])
    enable_cuda_graph: bool = True
    enable_torch_compile: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda:0"

    # KV Cache
    use_kv_cache: bool = False  # Enable prefix KV cache for decode iterations

    # Foundry Graph Pool
    use_foundry: bool = False               # Replace CudaGraphRunner with Foundry
    foundry_archive_dir: str = ""           # Path to Foundry graph archive
    enable_arbiter: bool = True             # Enable Confidence-Aware Arbiter (logging)

    # Starvation detection (APPC)
    starvation_threshold: int = 20  # iterations without progress before priority boost
