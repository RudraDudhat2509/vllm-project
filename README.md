# Beyond PagedAttention

Reproducing the memory/speed trade-off in vLLM's PagedAttention, and reproducing
the fix proposed by Microsoft Research's vAttention, at small scale in pure
PyTorch (CPU only, no GPU access at this stage).

ML project course (IIT Bhilai) proposal + working microbenchmark.

## The question

vLLM's [PagedAttention](https://arxiv.org/abs/2309.06180) stores each request's
KV cache in small fixed-size blocks instead of one contiguous buffer, the same
trick OS virtual memory uses for paging. That cuts KV-cache memory waste from
60-80% down to under 4%, so vLLM can batch far more requests at once.

That win isn't free. The vLLM paper's own numbers show its paged kernel is
20-26% slower than a non-paged kernel, because every read now has to walk a
block table instead of doing one contiguous memory access. Microsoft's
[vAttention](https://arxiv.org/abs/2405.04437) (ASPLOS'25) argues you can keep
the memory-saving property without paying that cost, by keeping the *virtual*
address space contiguous and only committing *physical* memory in blocks
underneath it (via CUDA's virtual memory APIs). It's live enough that there's
an [open GitHub issue on vllm-project/vllm](https://github.com/vllm-project/vllm/issues/17612)
about adopting it.

We don't have GPU access, so we can't reproduce the CUDA-level mechanism.
Instead we reproduce the *behavioral* trade-off in pure PyTorch: same
attention math, three different KV-cache backends, isolate exactly where the
slowdown comes from and whether decoupling virtual contiguity from physical
allocation actually recovers it.

## What's in this repo

```
src/
  attention.py   - shared decode-step attention math (identical across all 3 backends)
  caches.py       - NaiveContiguousCache, PagedCache (vLLM-style), VAttentionCache (ours)
  benchmark.py    - runs all 3 backends, times decode, tracks physical block usage
  plot.py         - turns results/benchmark_results.json into results/benchmark.png
results/
  benchmark_results.json  - raw numbers from the last run
  benchmark.png            - the figure used in the proposal
proposal/
  main.tex, ref.bib, fig1.png  - the IEEE-format project proposal (course template)
```

## Running it

```
pip install -r requirements.txt
cd src
python benchmark.py   # writes ../results/benchmark_results.json
python plot.py         # writes ../results/benchmark.png
```

## What we found (last run: 16 sequences, 4 layers, 8 heads, 256 decode steps, CPU)

| backend | mean step time | vs naive | physical memory |
|---|---|---|---|
| naive contiguous | 6.10 ms | baseline | commits everything up front |
| paged (vLLM-style) | 26.38 ms | **+332%** | grows on demand |
| vAttention-style (ours) | 6.54 ms | **+7.2%** | grows on demand, identical to paged |

The paged reproduction is slower than the paper's own reported 20-26%, because
we have no fused CUDA kernel, just a Python loop over blocks -- that gap is
called out explicitly in the proposal, not hidden. The direction holds: paging
costs real speed, and it's recoverable by decoupling virtual addressing from
physical allocation, which is exactly vAttention's actual claim.

## Team

- Rudra Dudhat (B24DS506) -- system design, implementation, benchmarking, proposal
- Mohak Arya (12341420) -- stretch-goal real-model integration, results verification
- Vatsal Yadav (B24DS036) -- stretch-goal real-model integration, presentation/viva

## AI disclosure

Claude (Claude Code CLI) was used for literature search/summarization, drafting
the initial implementation to the team's spec, and drafting the proposal text.
Every number in the proposal comes from `src/benchmark.py` actually being run,
not from the papers or from generated text. Full disclosure is in the
proposal's appendix.
