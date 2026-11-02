# Beyond PagedAttention

Reproducing the memory and speed trade-off in vLLM's PagedAttention, and reproducing the fix proposed by Microsoft Research's vAttention, at small scale in pure PyTorch, CPU only, no GPU access at this stage.

ML project course (IIT Bhilai) proposal and working microbenchmark.

## The question

vLLM's [PagedAttention](https://arxiv.org/abs/2309.06180) stores each request's KV cache in small fixed size blocks instead of one contiguous buffer, the same trick OS virtual memory uses for paging. That cuts KV cache memory waste from roughly 60 to 80 percent down to under 4 percent, so vLLM can batch far more requests at once.

That win is not free. The vLLM paper's own numbers show its paged kernel is 20 to 26 percent slower than a non-paged kernel, because every read now has to walk a block table instead of doing one contiguous memory access. Microsoft's [vAttention](https://arxiv.org/abs/2405.04437) (ASPLOS'25) argues you can keep the memory saving property without paying that cost, by keeping the virtual address space contiguous and only committing physical memory in blocks underneath it, using CUDA's virtual memory APIs. It is live enough that there is an [open GitHub issue on vllm-project/vllm](https://github.com/vllm-project/vllm/issues/17612) about adopting it.

We do not have GPU access, so we cannot reproduce the CUDA level mechanism. Instead we reproduce the behavioral trade-off in pure PyTorch: the same attention math, four different KV cache backends, isolating exactly where the slowdown comes from and whether decoupling virtual contiguity from physical allocation actually recovers it. The fourth backend is our own idea, not from either paper, see below.

## What is in this repo

```
src/attention.py    shared decode-step attention math, identical across all four backends
src/caches.py        NaiveContiguousCache, PagedCache (vLLM-style), VAttentionCache (ours), AdaptiveBlockCache (ours)
src/benchmark.py     runs all four backends, times decode, tracks physical block usage
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

Last run: 16 sequences, 4 layers, 8 heads, 256 decode steps, CPU only, pooled across 12 trials with rotated run order so machine load drift cannot unfairly penalize whichever backend happens to run last.

| backend | mean step time | vs naive | physical memory |
|---|---|---|---|
| naive contiguous | 7.73 ms | baseline | commits everything up front |
| paged (vLLM-style) | 29.40 ms | **+280% slower** | grows on demand, fixed block size |
| vAttention-style (ours) | 8.25 ms | within noise of naive | grows on demand, identical to paged |
| adaptive block (ours) | 9.47 ms | within noise of naive | grows on demand, adaptive chunk size, ~30% fewer blocks than paged or vAttention-style |

The paged reproduction is slower than the paper's own reported 20 to 26 percent, because we have no fused CUDA kernel, only a Python loop over blocks; that gap is called out explicitly in the proposal, not hidden. The direction holds: paging costs real speed, and it is recoverable by decoupling virtual addressing from physical allocation, which is exactly vAttention's claim.

`AdaptiveBlockCache` is our own idea: a single growing buffer per layer, like `VAttentionCache`, except nothing is preallocated to `max_seq_len` up front. It grows lazily in chunks, the same way a dynamic array doubles when it runs out of room: small chunks early in a sequence, bigger chunks once it survives past a threshold, the same pattern real allocators use for differently sized objects. We tried a version that kept PagedCache's per-sequence block list but shrank the block count; it was still slower than naive, so we dug into why instead of accepting the number. The real cause, isolated with a controlled microbenchmark, had nothing to do with block count: PagedCache stores every sequence in a separate Python list, so producing the single tensor the attention math needs forces a real memory copy on every decode step, no matter how few or how large the blocks are. `VAttentionCache` and `NaiveContiguousCache` avoid this because they store the whole batch in one tensor from the start, so a read is a free slice, not a copy. Once `AdaptiveBlockCache` was rebuilt on that same single-tensor structure, it landed in the same speed class as naive and vAttention-style (well within normal CPU noise) while still using about 30 percent fewer blocks than either, because its chunks adapt to sequence length instead of committing a fixed size every time. See `VIVA_PREP.md` for the full writeup, including the earlier version that didn't work and why.

## Team

| member | roll no | role |
|---|---|---|
| Rudra Dudhat | B24DS506 | system design, implementation of all three cache backends, benchmarking |
| Vatsal Yadav | B24DS036 | verification, evaluation metrics, validating results against the papers |
| Mohak Arya | 12341420 | literature review, research angle, written deliverables |

## AI disclosure

Claude (Claude Code CLI) was used for literature search and summarization, drafting the initial implementation to the team's spec, and drafting the proposal text. Every number in the proposal comes from `src/benchmark.py` actually being run, not from the papers and not from generated text. Full disclosure is in the proposal's appendix.
