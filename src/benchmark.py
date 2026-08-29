"""
Microbenchmark: decode a batch of sequences token-by-token through all three
KV-cache strategies, using identical random Q/K/V at every step so the only
variable is cache management. Measures:

  1. wall-clock time per decode step (speed)
  2. physical blocks committed over the run (memory)

This isolates the exact trade-off the vLLM paper reports (paging saves
memory but costs speed) and the exact fix vAttention claims (keep the
memory saving, recover the speed). No GPU, no real model weights --
this is a controlled systems microbenchmark, same spirit as how the
papers themselves report kernel-level numbers before full end-to-end runs.
"""

import json
import time
import torch

from attention import decode_step_attention
from caches import NaiveContiguousCache, PagedCache, VAttentionCache

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
    }

    results = {}
    for name, cls in strategies.items():
        print(f"running {name} ...")
        step_times, block_counts = run_decode(cls)
        total_time = sum(step_times)
        mean_step_time_ms = 1000 * total_time / len(step_times)
        results[name] = {
            "total_decode_time_s": total_time,
            "mean_step_time_ms": mean_step_time_ms,
            "final_physical_blocks": block_counts[-1],
            "block_counts_over_time": block_counts,
        }
        print(f"  mean step time: {mean_step_time_ms:.3f} ms | final physical blocks: {block_counts[-1]}")

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
