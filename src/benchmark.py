"""
Microbenchmark: decode a batch of sequences token-by-token through four
KV-cache strategies, using identical random Q/K/V at every step so the only
variable is cache management. Measures:

  1. wall-clock time per decode step (speed), mean and tail percentiles
  2. physical blocks committed over the run (memory)

This isolates the exact trade-off the vLLM paper reports (paging saves
memory but costs speed) and the exact fix vAttention claims (keep the
memory saving, recover the speed). The fourth strategy, AdaptiveBlockCache,
is our own idea, not from either paper: see caches.py for what it does and
why. No GPU, no real model weights; this is a controlled systems
microbenchmark, same spirit as how the papers themselves report
kernel-level numbers before full end-to-end runs.
"""

import json
import time
import torch

from attention import decode_step_attention
from caches import NaiveContiguousCache, PagedCache, VAttentionCache, AdaptiveBlockCache

torch.manual_seed(0)

BATCH_SIZE = 16
N_LAYERS = 4
N_HEADS = 8
HEAD_DIM = 64
MAX_SEQ_LEN = 256
BLOCK_SIZE = 16  # matches vLLM's default block size


def run_decode(cache_cls, warmup=10):
    cache = cache_cls(BATCH_SIZE, N_LAYERS, N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BLOCK_SIZE)
    step_times = []
    block_counts = []

    for t in range(MAX_SEQ_LEN):
        start = time.perf_counter()
        for layer in range(N_LAYERS):
            q = torch.randn(BATCH_SIZE, N_HEADS, 1, HEAD_DIM)
            k_new = torch.randn(BATCH_SIZE, N_HEADS, 1, HEAD_DIM)
            v_new = torch.randn(BATCH_SIZE, N_HEADS, 1, HEAD_DIM)
            k_hist, v_hist = cache.append_and_read(layer, t, k_new, v_new)
            _ = decode_step_attention(q, k_hist, v_hist)
        elapsed = time.perf_counter() - start
        if t >= warmup:  # drop warmup steps from timing stats
            step_times.append(elapsed)
        block_counts.append(cache.physical_blocks_used(t))

    return step_times, block_counts


def main():
    strategies = {
        "naive_contiguous": NaiveContiguousCache,
        "paged": PagedCache,
        "vattention_style": VAttentionCache,
        "adaptive_block": AdaptiveBlockCache,
    }
    names = list(strategies)
    n_trials = len(names)  # one trial per strategy, so each gets a turn
                            # running first, so no single strategy is always
                            # measured during the most loaded part of the run

    pooled_step_times = {name: [] for name in strategies}
    block_counts_by_name = {}

    for trial in range(n_trials):
        order = names[trial:] + names[:trial]  # rotate the starting position
        print(f"trial {trial + 1}/{n_trials}, order: {order}")
        for name in order:
            step_times, block_counts = run_decode(strategies[name])
            pooled_step_times[name].extend(step_times)
            block_counts_by_name[name] = block_counts  # deterministic, same every trial

    results = {}
    for name in strategies:
        step_times = pooled_step_times[name]
        block_counts = block_counts_by_name[name]
        total_time = sum(step_times)
        mean_step_time_ms = 1000 * total_time / len(step_times)
        times_ms = torch.tensor(step_times) * 1000
        p50, p95, p99 = torch.quantile(times_ms, torch.tensor([0.50, 0.95, 0.99]))
        results[name] = {
            "total_decode_time_s": total_time,
            "mean_step_time_ms": mean_step_time_ms,
            "p50_step_time_ms": p50.item(),
            "p95_step_time_ms": p95.item(),
            "p99_step_time_ms": p99.item(),
            "final_physical_blocks": block_counts[-1],
            "block_counts_over_time": block_counts,
        }
        print(f"  mean step time: {mean_step_time_ms:.3f} ms | p95: {p95:.3f} ms | p99: {p99:.3f} ms | final physical blocks: {block_counts[-1]}")

    baseline_ms = results["naive_contiguous"]["mean_step_time_ms"]
    for name, r in results.items():
        r["slowdown_vs_naive_pct"] = 100 * (r["mean_step_time_ms"] - baseline_ms) / baseline_ms

    with open("../results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nsummary (slowdown vs naive contiguous baseline):")
    for name, r in results.items():
        print(f"  {name:20s} {r['slowdown_vs_naive_pct']:+.1f}%  (final physical blocks: {r['final_physical_blocks']})")


if __name__ == "__main__":
    main()
