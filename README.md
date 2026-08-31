# Beyond PagedAttention

Reproducing the memory and speed trade-off in vLLM's PagedAttention, and reproducing the fix proposed by Microsoft Research's vAttention, at small scale in pure PyTorch, CPU only, no GPU access at this stage.

ML project course (IIT Bhilai) proposal and working microbenchmark.

## The question

vLLM's [PagedAttention](https://arxiv.org/abs/2309.06180) stores each request's KV cache in small fixed size blocks instead of one contiguous buffer, the same trick OS virtual memory uses for paging. That cuts KV cache memory waste from roughly 60 to 80 percent down to under 4 percent, so vLLM can batch far more requests at once.

That win is not free. The vLLM paper's own numbers show its paged kernel is 20 to 26 percent slower than a non-paged kernel, because every read now has to walk a block table instead of doing one contiguous memory access. Microsoft's [vAttention](https://arxiv.org/abs/2405.04437) (ASPLOS'25) argues you can keep the memory saving property without paying that cost, by keeping the virtual address space contiguous and only committing physical memory in blocks underneath it, using CUDA's virtual memory APIs. It is live enough that there is an [open GitHub issue on vllm-project/vllm](https://github.com/vllm-project/vllm/issues/17612) about adopting it.

We do not have GPU access, so we cannot reproduce the CUDA level mechanism. Instead we reproduce the behavioral trade-off in pure PyTorch: the same attention math, three different KV cache backends, isolating exactly where the slowdown comes from and whether decoupling virtual contiguity from physical allocation actually recovers it.

## What is in this repo

```
src/attention.py    shared decode-step attention math, identical across all three backends
src/caches.py        NaiveContiguousCache, PagedCache (vLLM-style), VAttentionCache (ours)
src/benchmark.py     runs all three backends, times decode, tracks physical block usage
src/plot.py           turns results/benchmark_results.json into results/benchmark.png
results/benchmark_results.json   raw numbers from the last run
results/benchmark.png             the figure used in the proposal
proposal/main.tex, ref.bib, fig1.png   the IEEE format project proposal (course template)
```

## Running it

```
pip install -r requirements.txt
cd src
python benchmark.py   # writes ../results/benchmark_results.json
python plot.py          # writes ../results/benchmark.png
```

## What we found

Last run: 16 sequences, 4 layers, 8 heads, 256 decode steps, CPU only.

| backend | mean step time | vs naive | physical memory |
|---|---|---|---|
| naive contiguous | 6.10 ms | baseline | commits everything up front |
| paged (vLLM-style) | 26.38 ms | **332% slower** | grows on demand |
| vAttention-style (ours) | 6.54 ms | **7.2% slower** | grows on demand, identical to paged |

The paged reproduction is slower than the paper's own reported 20 to 26 percent, because we have no fused CUDA kernel, only a Python loop over blocks; that gap is called out explicitly in the proposal, not hidden. The direction holds: paging costs real speed, and it is recoverable by decoupling virtual addressing from physical allocation, which is exactly vAttention's claim.

## Team

| member | roll no | role |
|---|---|---|
| Rudra Dudhat | B24DS506 | literature review, project scaffold, most of the coding |
| Vatsal Yadav | B24DS036 | verifying results, further experiments for visualizations |
| Mohak Arya | 12341420 | ideation, preparing deliverables such as the proposal |

## AI disclosure

Claude (Claude Code CLI) was used for literature search and summarization, drafting the initial implementation to the team's spec, and drafting the proposal text. Every number in the proposal comes from `src/benchmark.py` actually being run, not from the papers and not from generated text. Full disclosure is in the proposal's appendix.
