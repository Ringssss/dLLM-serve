# SpecDiff: Speculative Convergence Scheduling for dLLM Serving

## 核心洞察

CW-SRPT 在高负载时通过调度优化尾延迟（-62% TTFT P90），但在低负载时无效——因为没有调度空间。

**关键问题：低负载时 GPU 在等请求，能不能用这段空闲算力来加速单请求的去噪收敛？**

AR LLM 无法做到这一点——每次 forward 只产出 1 token，没法"投机性地多算"。但 dLLM 可以：

## SpecDiff 设计

### 核心机制：Speculative Multi-Path Denoising

低负载时（batch < max_batch），将空闲 batch slot 用于**同一请求的投机并行去噪路径**：

```
传统 dLLM (bs=1 时)：
  iter 1: [req_A block tokens] → unmask 3/32 tokens
  iter 2: [req_A block tokens] → unmask 2/32 tokens
  iter 3: [req_A block tokens] → unmask 4/32 tokens
  ...共 ~12 iter 才完成 1 个 block

SpecDiff (bs=1 但有 8 个 slot)：
  iter 1: [req_A_path1, req_A_path2, ..., req_A_path8] → 8 条并行去噪路径
          path1 用 threshold=0.9 (标准)
          path2 用 threshold=0.7 (更激进)
          path3 用 threshold=0.5 (非常激进)
          ...
  → 在 1-2 iter 内，某条路径完全收敛
  → 选择质量最好的收敛路径作为输出
  → 单 block 从 ~12 iter 降到 ~2-3 iter = 4-6x 加速!
```

### 为什么这在 AR 中不可能？

AR LLM 的每个 token 依赖前一个 token——无法并行尝试多条路径。
dLLM 的 block 内所有 token 是**独立并行**去噪的——天然支持多路径。

### 两种模式的统一

```
┌─────────────────────────────────────────────────────┐
│                  SpecDiff Engine                      │
│                                                       │
│  if active_reqs >= max_batch:                        │
│    → CW-SRPT mode (调度优化)                          │
│    → confidence-weighted batch selection              │
│    → same as v2, proven -62% TTFT P90               │
│                                                       │
│  if active_reqs < max_batch:                         │
│    → Speculative mode (单请求加速)                     │
│    → fill empty slots with speculative paths          │
│    → multi-threshold parallel denoising               │
│    → pick best converged path                         │
│    → 4-6x single-request speedup                     │
│                                                       │
│  transition: smooth, automatic, per-iteration         │
└─────────────────────────────────────────────────────┘
```

### 具体实现

#### Speculative Path Generation

```python
def fill_speculative_paths(batch, max_batch, config):
    """Fill empty batch slots with speculative paths for active req""
    empty_slots = max_batch - len(batch)
    if empty_slots <= 0:
        return batch  # 满载，走 CW-SRPT

    # 选择最有投机价值的请求：n_masked 最多的（收敛最远的）
    candidates = sorted(batch, key=lambda r: -r.n_masked)

    spec_paths = []
    thresholds = [0.7, 0.5, 0.3, 0.1]  # 递减阈值 = 更激进的 unmask

    for slot_idx in range(empty_slots):
        # Round-robin 分配投机路径给最需要的请求
        target = candidates[slot_idx % len(candidates)]
        th = thresholds[min(slot_idx, len(thresholds) - 1)]

        # 创建投机 clone：复制当前 token 状态
        spec_req = clone_request_for_speculation(target, th)
        spec_paths.append((target.id, spec_req))

    return batch + [s[1] for s in spec_paths]
```

#### Speculative Commit

每次 forward 后，检查投机路径是否已经收敛：

```python
def commit_speculative_results(batch, spec_paths, original_reqs):
    """If a speculative path fully converged, adopt it."""
    for original_id, spec_req in spec_paths:
        if spec_req.n_masked == 0:  # 完全收敛！
            original = find_by_id(original_reqs, original_id)
            # 验证质量：spec_req 的 output 与 original 的高 confidence tokens 一致
            if quality_check_passed(original, spec_req):
                original.x = spec_req.x  # 采纳投机结果
                original.advance_block()
                # 跳过了 ~10 iter！
```

#### Quality Check

投机路径用更低的 threshold，可能 unmask 低 confidence token → 质量风险。
验证：比较投机路径与原路径的**高 confidence 位置**是否一致。

```python
def quality_check_passed(original, speculative):
    """Check that speculative tokens agree with high-confidence original tokens."""
    orig_block = original.x[original.block_start:original.block_end]
    spec_block = speculative.x[speculative.block_start:speculative.block_end]

    # 已在原路径中 unmask 的 token（高 confidence）应该和投机路径一致
    unmasked = orig_block != MASK_ID
    if unmasked.any():
        agreement = (orig_block[unmasked] == spec_block[unmasked]).float().mean()
        return agreement > 0.95  # 95% 一致
    return True  # 原路径还没 unmask 任何 token，无法验证
```

### 预期收益

#### 低负载 (1-3 active reqs)
- 传统：12 iter/block × 4 blocks = 48 fwds/req, ~430ms
- SpecDiff：3 iter/block × 4 blocks = 12 fwds/req, ~110ms
- **~4x 单请求加速, -75% E2E latency**

#### 高负载 (8+ active reqs)
- CW-SRPT 模式，与 v2 相同
- **-62% TTFT P90 vs FCFS**

#### 中等负载 (4-7 active reqs)
- 混合：部分 slot 做真实请求，部分做投机
- **-30-50% E2E latency vs 纯 CW-SRPT**

### 与 AR Serving 的本质区别

| | AR (FastServe MLFQ) | dLLM (SpecDiff) |
|---|---|---|
| 低负载优化 | 无（等请求） | **投机多路径去噪** |
| 高负载优化 | MLFQ 优先级队列 | **CW-SRPT confidence 调度** |
| 空闲 slot 利用 | 不可能 | **投机填充** |
| Preemption | 昂贵（KV swap） | **免费** |
| 工作量预测 | Profile-based（离线） | **Confidence-based（在线实时）** |
| 核心信号 | 排队时间 | **Per-token confidence** |

### 论文 Story

> dLLM 的 iterative denoising 带来了 AR LLM 不具备的运行时可观测性：
> per-token confidence。SpecDiff 利用这一信号实现两层优化：
>
> 1. **高负载**：CW-SRPT 用 confidence 预测剩余工作量，
>    优化调度顺序 → -62% TTFT P90
>
> 2. **低负载**：Speculative Multi-Path Denoising 用空闲 batch slot
>    并行尝试不同 threshold 的去噪路径 → 4x 单请求加速
>
> 这是 dLLM-native 的调度范式：confidence 既是调度信号，
> 也是投机验证的质量门控。AR 系统无法复制。

### 实现优先级

1. **Phase 1**: Speculative path generation + commit (核心机制)
2. **Phase 2**: Quality check + 自动 threshold 搜索
3. **Phase 3**: CW-SRPT ↔ Speculative 平滑切换
4. **Phase 4**: Foundry dense graph 支持投机路径的 variable bs
