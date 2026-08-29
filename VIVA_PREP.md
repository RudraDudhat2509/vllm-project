# Viva prep

Read this until you can answer every question without looking. Plain English
first, the technical term comes right after it, so you can say either version
depending on how the question is phrased.

---

**Q: What is a KV cache and why does it exist?**

When a transformer generates text one word at a time, it doesn't want to
redo the full attention computation over every previous word at every step
-- that would be O(n^2) work repeated n times. So for every previous token,
it saves the "key" and "value" vectors computed for it (that's the K and V
in "KV cache") and just reuses them. Generating token 500 means attending
over the saved K/V of tokens 1-499 plus computing K/V fresh only for token
500. This cache is what makes autoregressive generation fast -- but it also
means memory usage grows with every token you generate, per request, and
you don't know in advance how long a request will run.

**Q: What problem does PagedAttention actually solve?**

Before vLLM, serving engines had to guess a max length and reserve that
much contiguous GPU memory for every request's KV cache up front, like
booking a hotel's entire top floor for a guest just in case they need it,
even if they only stay one night. Most of that memory sits empty
(fragmentation), so you can't fit many requests in a batch, and batch size
is what determines GPU throughput. PagedAttention instead hands out memory
in small fixed-size blocks (16 tokens' worth, by default) only as a
sequence actually grows, the same idea as OS virtual memory paging. Memory
waste drops from 60-80% to under 4%, so far more requests fit in a batch.

**Q: So what's the catch — why does this project exist?**

Blocks aren't stored next to each other in memory (non-contiguous). To read
a request's full KV history now, the kernel has to look up which physical
block holds each chunk (the "block table") and jump around memory instead
of reading one straight line. That lookup and jumping is real work on every
single decode step. The vLLM paper's own numbers say this makes their kernel
20-26% slower than a non-paged one. Microsoft's vAttention paper (2024) says
you don't have to accept that cost -- you can keep the *logical/virtual*
address space one contiguous line (so reads are simple and fast) while still
only committing *physical* GPU memory in small blocks underneath it, using
CUDA's virtual-memory system calls. Virtual addresses are just numbers you
promise to use later; physical memory is the actual chips being used. You
can reserve a huge range of virtual addresses for free — it costs you
nothing until you actually write real data into it, which is when physical
memory gets committed. vAttention exploits exactly that gap.

**Q: What did you actually build, and why no GPU?**

We don't have GPU access at the proposal stage, and vAttention's real trick
uses CUDA-specific virtual memory calls that don't exist on CPU. So we built
a controlled behavioral analogy in pure PyTorch: three KV-cache backends
sharing one identical attention function, so cache management is the only
variable.
- `NaiveContiguousCache` — pre-allocates full max length up front. Fast reads,
  wastes memory (the pre-vLLM approach).
- `PagedCache` — allocates memory in 16-token blocks on demand, but has to
  gather+concatenate every block on every read. This is our reproduction of
  PagedAttention's actual mechanism, minus the CUDA kernel fusion.
- `VAttentionCache` — same on-demand block accounting for memory (so the
  memory-saving claim is preserved), but reads go through one contiguous
  buffer we extend by a single token each step, instead of re-gathering
  everything. This is the *behavioral effect* of vAttention's fix, not its
  actual CUDA mechanism — say that clearly if asked, don't overclaim it.

**Q: What did you measure, and what did you find?**

16 sequences, 4 layers, 8 heads, 64-dim heads, 256 decode steps, identical
random Q/K/V every step across all three backends so timing differences come
only from cache management, not from what's being computed.
- Paged: **332% slower** per decode step than the naive contiguous baseline.
- Our vAttention-style version: only **7.2% slower** — it recovers almost all
  of that gap.
- Physical memory: naive commits everything immediately from step 0; paged
  and vAttention-style grow identically, block by block, on demand (their
  memory curves overlap exactly by construction).

**Q: Why is your 332% so much bigger than the paper's 20-26%?**

Because the paper compares two real, hand-optimized fused CUDA kernels. Our
paged backend is a plain Python `for` loop over blocks calling `torch.cat`
every step — there's no kernel fusion, so the relative overhead of "doing
several small operations" versus "doing one big operation" is naturally much
larger in an unoptimized interpreted loop than in compiled, fused GPU code.
We say this directly in the proposal's Limitations section instead of
letting the number stand without context. The *direction* of the result
(paging costs real speed; decoupling virtual/physical recovers it) is what
we're claiming, not the exact magnitude.

**Q: What's the block size trade-off, and why 16?**

Smaller blocks (say, 4 tokens) mean less wasted memory at the tail end of a
sequence (less "internal fragmentation" — the last block is rarely full),
but more blocks total means more block-table entries to track and more
per-read overhead. Bigger blocks mean less bookkeeping overhead but more
wasted memory per sequence. 16 is vLLM's own default, chosen empirically as
a reasonable middle point on real GPUs; we reused it so our numbers are
comparable to the real system rather than an arbitrary choice.

**Q: Why only the decode phase, not prefill?**

Prefill (processing the initial prompt) computes attention over a big batch
of tokens all at once — one large contiguous matrix operation regardless of
how the cache is laid out, because you're writing the whole prompt's KV in
one shot. Decode (generating one new token at a time) is where the cache
gets read and appended to repeatedly, one token per step, which is exactly
where block-table lookups happen over and over and the non-contiguous
layout actually costs you something. That's why both the vLLM ablation and
our benchmark focus there.

**Q: Where's the honesty in this project — what would you NOT claim?**

- Not claiming to beat vLLM or vAttention's real numbers — this is a
  small-scale mechanism check, not a production benchmark.
- Not claiming our "physical memory saved" number is measured RAM — it's an
  analytical block count, the same convention the papers themselves use to
  report memory savings, because Python can't do real lazy physical-page
  commit the way CUDA can.
- Not claiming vAttention-style is a CUDA virtual memory reproduction — it's
  a behavioral emulation of the same idea, clearly labeled as such.
- Not claiming any result about model output quality — every experiment
  runs on random tensors, no real language model.

**Q: What's next if this gets approved for full implementation?**

Get GPU access (Kaggle's free 30 hrs/week or Colab), wire the same three
backends (or vAttention's real CUDA VMM implementation) into a small real
model like GPT-2 or TinyLlama, and re-run the same benchmark methodology
end-to-end with real prompts (ShareGPT-style, same source vLLM's own paper
used) to see if the same relative pattern (paging costs speed, virtual/
physical decoupling recovers it) survives outside a CPU microbenchmark.
