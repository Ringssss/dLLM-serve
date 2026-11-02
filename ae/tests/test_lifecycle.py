from array import array
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

from sglang.srt.dllm.algorithm.cw_srpt_v3 import CW_SRPT_V3
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin, DllmManager


def config(algorithm='CW_SRPT_V3'):
    return DllmConfig(algorithm, {'target_iters': 2, 'min_conf': .6,
                                'early_break_ratio': .75}, 4, 9, 4)


class Request(ReqDllmMixin):
    def __init__(self, tokens):
        self.origin_input_ids = array('q')
        self.output_ids = array('q')
        self.fill_ids = array('q', tokens)
        self.prefix_indices = torch.empty(0, dtype=torch.long)
        self.init_diffusion_llm(config())
        self.update_finish_state = Mock()

    def finished(self):
        return False


def writeback(requests, blocks, logits):
    sched = NS(token_to_kv_pool_allocator=Mock(), output_streamer=Mock(),
               metrics_reporter=NS(num_generated_tokens=0, report_prefill_stats=Mock()))
    batch = NS(reqs=requests, batch_size=lambda: len(requests), return_logprob=False,
               prefill_stats=None, dp_cooperation_info=None)
    result = NS(copy_done=None, next_token_ids=blocks, logits_output=logits, can_run_cuda_graph=False)
    SchedulerDllmMixin.process_batch_result_dllm(sched, batch, result)
    return sched


def test_partial_return_resumes_without_publishing_masks():
    algo = CW_SRPT_V3(config())
    batch = NS(batch_size=4, input_ids=torch.full((16,), 9, dtype=torch.long))
    logits = torch.zeros(4, 4, 10)
    logits[:3, :, 1] = 10  # three easy blocks
    logits[3, :, 1] = .1   # hard block advances only through fallback
    runner = NS(forward=lambda *a, **kw: NS(logits_output=NS(full_logits=logits.reshape(16, 10)), can_run_graph=False))
    out, blocks, _ = algo.run(runner, batch)
    assert 9 in blocks[-1].tolist(), 'Fixture must exercise early return'
    reqs = [Request([9] * 4) for _ in range(4)]
    sched = writeback(reqs, blocks, out)
    assert sched.metrics_reporter.num_generated_tokens == 12
    assert list(reqs[-1].output_ids) == []
    assert reqs[-1].dllm_partial
    saved = list(reqs[-1].fill_ids)
    reqs[-1]._init_fill_ids_for_dllm()
    assert list(reqs[-1].fill_ids) == saved
    assert reqs[-1].dllm_block_offset == 0
    next_batch = NS(batch_size=1, input_ids=torch.tensor(saved))
    next_runner = NS(forward=lambda *a, **kw: NS(logits_output=NS(full_logits=logits[3]), can_run_graph=False))
    out2, blocks2, _ = algo.run(next_runner, next_batch)
    writeback([reqs[-1]], blocks2, out2)
    assert list(reqs[-1].output_ids) == [1] * 4
    assert not reqs[-1].dllm_partial


def test_noncontiguous_holes_and_prompt_prefix_are_not_duplicated():
    req = Request([3, 9, 2, 9])
    req.origin_input_ids = array('q', [3])
    writeback([req], [torch.tensor([3, 1, 2, 1])], NS())
    assert list(req.output_ids) == [1, 2, 1]


def test_baseline_keeps_fifo_order():
    manager = DllmManager(config('LowConfidence'))
    requests = [NS(is_dllm_prefill=lambda: False, dllm_remaining_work=w) for w in [9, 1, 4]]
    manager.waiting_queue = requests.copy()
    assert manager.get_decode_requests() == requests
