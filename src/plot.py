"""Reads results/benchmark_results.json and produces results/benchmark.png."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("../results/benchmark_results.json") as f:
    results = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

names = ["naive_contiguous", "paged", "vattention_style"]
labels = ["Naive\ncontiguous", "Paged\n(vLLM-style)", "vAttention-style\n(ours)"]
colors = ["#888888", "#d62728", "#2ca02c"]
step_times = [results[n]["mean_step_time_ms"] for n in names]

ax1.bar(labels, step_times, color=colors)
ax1.set_ylabel("mean decode step time (ms)")
ax1.set_title("Speed: per-step decode latency")
for i, v in enumerate(step_times):
    ax1.text(i, v + max(step_times) * 0.02, f"{v:.2f} ms", ha="center", fontsize=9)

for n, c, lbl in zip(names, colors, labels):
    counts = results[n]["block_counts_over_time"]
    ax2.plot(range(len(counts)), counts, label=lbl.replace("\n", " "), color=c)
ax2.set_xlabel("decode step (token position)")
ax2.set_ylabel("physical KV-cache blocks committed")
ax2.set_title("Memory: physical blocks over time")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("../results/benchmark.png", dpi=150)
print("saved ../results/benchmark.png")
