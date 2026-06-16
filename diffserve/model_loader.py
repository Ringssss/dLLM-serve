"""
Model loading and SGLang bootstrap for DiffServe.

Extracts the SGLang-dependent initialization from dInfer's test scripts
into a reusable module. This includes:
  - vLLM mock setup (required by dInfer model code)
  - dinfer package module loading
  - SGLang distributed initialization
  - Model weight loading
  - ModelRunner creation with CUDA Graph capture
"""

import sys
import os
import types
import importlib
import logging

import torch
import numpy as np

from .config import DiffServeConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# vLLM Mock Setup
# ═══════════════════════════════════════════════════════════════════

def setup_vllm_mocks():
    """Install vLLM mock modules so dInfer's model code can import them.

    dInfer's modeling_llada2_moe_sglang.py imports vLLM linear/layernorm layers.
    Since we use SGLang's actual kernels (not vLLM's), we mock these interfaces.
    """
    if 'vllm' in sys.modules:
        return  # already set up

    vllm_mock = types.ModuleType('vllm')
    submods = [
        'distributed', 'config', 'forward_context', 'model_executor',
        'model_executor.layers', 'model_executor.layers.fused_moe',
        'model_executor.layers.fused_moe.layer',
        'model_executor.layers.linear', 'model_executor.layers.layernorm',
        'model_executor.models', 'model_executor.models.utils',
    ]
    for submod in submods:
        sys.modules[f'vllm.{submod}'] = types.ModuleType(f'vllm.{submod}')
    sys.modules['vllm'] = vllm_mock

    # Config mocks
    sys.modules['vllm.config'].ParallelConfig = type(
        'P', (), {'__init__': lambda s, **kw: None})
    sys.modules['vllm.config'].VllmConfig = type(
        'V', (), {'__init__': lambda s, **kw: None})
    sys.modules['vllm.config'].set_current_vllm_config = (
        lambda *a, **kw: type('c', (), {
            '__enter__': lambda s: None, '__exit__': lambda s, *a: None})())
    sys.modules['vllm.config'].get_current_vllm_config = lambda: None

    # Forward context mock
    sys.modules['vllm.forward_context'].set_forward_context = (
        lambda *a, **kw: type('c', (), {
            '__enter__': lambda s: None, '__exit__': lambda s, *a: None})())

    # Distributed mocks
    sys.modules['vllm.distributed'].get_tensor_model_parallel_rank = lambda: 0
    sys.modules['vllm.distributed'].get_tensor_model_parallel_world_size = lambda: 1
    sys.modules['vllm.distributed'].divide = lambda a, b: a // b
    sys.modules['vllm.distributed'].tensor_model_parallel_all_reduce = lambda t: t
    sys.modules['vllm.distributed'].tensor_model_parallel_all_gather = lambda t: t
    sys.modules['vllm.distributed'].split_tensor_along_last_dim = lambda t, n: torch.chunk(t, n, dim=-1)
    sys.modules['vllm.distributed'].EplbState = type('E', (), {})

    # Layer mocks
    sys.modules['vllm.model_executor.layers.fused_moe'].FusedMoE = type(
        'F', (torch.nn.Module,),
        {'__init__': lambda s, **kw: torch.nn.Module.__init__(s)})
    sys.modules['vllm.model_executor.layers.fused_moe'].fused_moe = lambda *a, **kw: None
    sys.modules['vllm.model_executor.layers.linear'].ColumnParallelLinear = torch.nn.Linear
    sys.modules['vllm.model_executor.layers.linear'].RowParallelLinear = torch.nn.Linear
    sys.modules['vllm.model_executor.layers.linear'].QKVParallelLinear = torch.nn.Linear
    sys.modules['vllm.model_executor.layers.linear'].ReplicatedLinear = torch.nn.Linear
    sys.modules['vllm.model_executor.layers.layernorm'] = types.ModuleType(
        'vllm.model_executor.layers.layernorm')
    sys.modules['vllm.model_executor.layers.layernorm'].rms_norm = lambda x, w, e: x

    # Models utils mock
    sys.modules['vllm.model_executor.models.utils'] = types.ModuleType(
        'vllm.model_executor.models.utils')
    sys.modules['vllm.model_executor.models.utils'].maybe_prefix = lambda *a: ''

    # deep_ep mock (for MoE expert parallel)
    if 'deep_ep' not in sys.modules:
        dep = types.ModuleType('deep_ep')
        dep.__spec__ = importlib.util.spec_from_loader('deep_ep', loader=None)
        dep.__path__ = []
        dep.Buffer = type('B', (), {
            'get_dispatch_config': staticmethod(lambda *a, **kw: None),
            'get_combine_config': staticmethod(lambda *a, **kw: None),
        })
        dep.Config = type('C', (), {})
        dep.EventOverlap = type('EO', (), {})
        sys.modules['deep_ep'] = dep


# ═══════════════════════════════════════════════════════════════════
# dinfer Module Loading
# ═══════════════════════════════════════════════════════════════════

DINFER_BASE = '/home/zhujianian/dInfer/python/dinfer'


def setup_dinfer_modules():
    """Register the dinfer package modules for import.

    This replicates the module loading pattern used in the test scripts
    to make dinfer importable without a proper pip install.
    """
    if 'dinfer' in sys.modules:
        return

    # Add to path
    dinfer_parent = os.path.dirname(DINFER_BASE)
    if dinfer_parent not in sys.path:
        sys.path.insert(0, dinfer_parent)

    dinfer_pkg = types.ModuleType('dinfer')
    dinfer_pkg.__path__ = [DINFER_BASE]
    dinfer_pkg.__package__ = 'dinfer'
    sys.modules['dinfer'] = dinfer_pkg

    for sub in ['model', 'decoding']:
        m = types.ModuleType(f'dinfer.{sub}')
        m.__path__ = [f'{DINFER_BASE}/{sub}']
        sys.modules[f'dinfer.{sub}'] = m
        setattr(dinfer_pkg, sub, m)

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load('dinfer.decoding.utils', f'{DINFER_BASE}/decoding/utils.py')
    _load('dinfer.decoding.parallel_strategy',
          f'{DINFER_BASE}/decoding/parallel_strategy.py')
    _load('dinfer.decoding.generate_uniform',
          f'{DINFER_BASE}/decoding/generate_uniform.py')
    _load('dinfer.model.modeling_llada2_moe_sglang',
          f'{DINFER_BASE}/model/modeling_llada2_moe_sglang.py')
    _load('dinfer.decoding.diffusion_runner',
          f'{DINFER_BASE}/decoding/diffusion_runner.py')


# ═══════════════════════════════════════════════════════════════════
# SGLang Distributed Initialization
# ═══════════════════════════════════════════════════════════════════

def init_sglang_distributed(device: str = "cuda:0"):
    """Initialize SGLang's distributed environment (single-GPU mode).

    This sets up NCCL, model parallelism, dp_attention, and MoE config
    required by dInfer's ModelRunner and model code.
    """
    from sglang.srt import distributed

    if not torch.distributed.is_initialized():
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12399')
        distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
        distributed.initialize_model_parallel(1, 1, 1, backend='nccl')


def _create_server_args(config: DiffServeConfig):
    """Create SGLang ServerArgs for model initialization."""
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(
        model_path=config.model_path,
        enable_dp_attention=True,
        trust_remote_code=True,
        tp_size=1,
        dp_size=1,
        pp_size=1,
    )
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
        set_global_server_args_for_scheduler(server_args)
    except ImportError:
        pass
    return server_args


# ═══════════════════════════════════════════════════════════════════
# Model + ModelRunner Creation
# ═══════════════════════════════════════════════════════════════════

def create_model_runner(config: DiffServeConfig):
    """Initialize model, tokenizer, and ModelRunner with CUDA Graph.

    Args:
        config: DiffServe configuration.

    Returns:
        Tuple of (model_runner, tokenizer).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # 1. Setup mocks and modules
    setup_vllm_mocks()
    setup_dinfer_modules()

    # 2. Set device
    device = torch.device(config.device)
    torch.cuda.set_device(device)

    # 3. Load tokenizer and model config
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(
        config.model_path, trust_remote_code=True)

    # 4. Initialize SGLang distributed
    init_sglang_distributed(config.device)

    # 5. Initialize SGLang components
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    server_args = _create_server_args(config)
    initialize_dp_attention(server_args=server_args, model_config=model_config)
    initialize_moe_config(server_args)

    # 6. Load model
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM

    logger.info(f"Loading model from {config.model_path}...")
    model = LLaDA2SGLangLM(config=model_config, expert_map_path='.').eval()
    torch.set_default_dtype(torch.bfloat16)
    model.load_weights(config.model_path, device=device)
    initialize_moe_config(server_args)
    model = model.to(device)
    model.after_processing()

    # 7. Foundry integration: patch CudaGraphRunner before ModelRunner init
    from dinfer.decoding.diffusion_runner import ModelRunner

    if config.use_foundry:
        _patch_cuda_graph_runner_for_foundry(config)

    # Dense batch sizes when Foundry enabled: [1..max_bs] instead of [1,2,4,8]
    bs_list = list(config.supported_batch_sizes)
    if config.use_foundry:
        max_bs = max(bs_list)
        bs_list = list(range(1, max_bs + 1))

    model_runner = ModelRunner(
        model,
        device,
        server_args=server_args,
        max_length=config.max_seq_length,
        block_length=config.block_length,
        prefill_lengths=config.prefill_lengths,
        enable_cuda_graph=config.enable_cuda_graph,
        supported_batch_sizes=bs_list,
        use_cross_block=True,
        enable_compile=config.enable_torch_compile,
    )

    # Save Foundry archive after capture (one-time, for instant reload later)
    if config.use_foundry and config.foundry_archive_dir:
        _save_foundry_archive(model_runner, config)

    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    logger.info(f"Model loaded. GPU memory: {mem_gb:.1f} GB")

    return model_runner, tokenizer


# ═══════════════════════════════════════════════════════════════════
# Foundry Integration
# ═══════════════════════════════════════════════════════════════════

_foundry_patched = False


def _patch_cuda_graph_runner_for_foundry(config: DiffServeConfig):
    """Replace torch.cuda.CUDAGraph with foundry.CUDAGraph in CudaGraphRunner.

    This gives us:
    1. Template-based topology sharing → faster capture
    2. save()/load() → persist graphs to disk for instant cold start
    3. Dense bs coverage → zero padding waste from batch sizing
    """
    global _foundry_patched
    if _foundry_patched:
        return

    try:
        from foundry.graph import CUDAGraph as FoundryCUDAGraph
    except ImportError:
        logger.warning("[Foundry] Not available — using standard torch CUDA graphs")
        return

    from dinfer.decoding.diffusion_runner import CudaGraphRunner

    def _foundry_create_device_graph(self):
        return FoundryCUDAGraph()

    CudaGraphRunner._create_device_graph = _foundry_create_device_graph
    _foundry_patched = True
    logger.info("[Foundry] Patched CudaGraphRunner → foundry.CUDAGraph (dense bs, save/load)")


def _save_foundry_archive(model_runner, config: DiffServeConfig):
    """Save captured Foundry graphs to disk for instant reload next startup."""
    archive_dir = config.foundry_archive_dir
    if not archive_dir:
        return

    try:
        import foundry
    except ImportError:
        return

    import os
    if os.path.exists(os.path.join(archive_dir, "manifest.json")):
        logger.info(f"[Foundry] Archive exists at {archive_dir}, skip save")
        return

    os.makedirs(archive_dir, exist_ok=True)
    runner = model_runner.graph_runner
    if runner is None:
        return

    import time
    t0 = time.perf_counter()
    idx = 0
    for key, graph in runner.graphs.items():
        bs, is_decode, length, cache_length = key
        phase = 'd' if is_decode else 'p'
        fname = f"graph_{idx}_bs{bs}_{phase}_l{length}_c{cache_length}.json"
        path = os.path.join(archive_dir, fname)
        try:
            out = runner.output_buffers[key]
            out_tensor = out.logits if hasattr(out, 'logits') else out
            graph.save(path, out_tensor)
            idx += 1
        except Exception as e:
            logger.warning(f"[Foundry] Save failed for {key}: {e}")

    try:
        foundry.save_graph_manifest(archive_dir)
    except Exception as e:
        logger.warning(f"[Foundry] Manifest save failed: {e}")

    elapsed = time.perf_counter() - t0
    logger.info(f"[Foundry] Saved {idx} graphs to {archive_dir} in {elapsed:.1f}s")
