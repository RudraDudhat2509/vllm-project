# Viva prep

Read this until you can answer every question without looking. Plain English
first, the technical term comes right after it, so you can say either version
depending on how the question is phrased.

---

**Q: What is a KV cache and why does it exist?**

When a transformer generates text one word at a time, it does not want to
redo the full attention computation over every previous word at every step:
that would be O(n^2) work repeated n times. So for every previous token,
it saves the "key" and "value" vectors computed for it, that is the K and V
in "KV cache", and just reuses them. Generating token 500 means attending
over the saved K/V of tokens 1 through 499 plus computing K/V fresh only for
token 500. This cache is what makes autoregressive generation fast, but it
also means memory usage grows with every token you generate, per request,
and you do not know in advance how long a request will run.

**Q: What problem does PagedAttention actually solve?**

Before vLLM, serving engines had to guess a max length and reserve that
much contiguous GPU memory for every request's KV cache up front, like
booking a hotel's entire top floor for a guest just in case they need it,
even if they only stay one night. Most of that memory sits empty, this is
fragmentation, so you cannot fit many requests in a batch, and batch size
is what determines GPU throughput. PagedAttention instead hands out memory
in small fixed size blocks, 16 tokens' worth by default, only as a
sequence actually grows, the same idea as OS virtual memory paging. Memory
waste drops from roughly 60 to 80 percent down to under 4 percent, so far
more requests fit in a batch.

**Q: So what is the catch, why does this project exist?**

Blocks are not stored next to each other in memory, this is what
non-contiguous means. To read a request's full KV history now, the kernel
has to look up which physical block holds each chunk, the "block table",
and jump around memory instead of reading one straight line. That lookup
and jumping is real work on every single decode step. The vLLM paper's own
numbers say this makes their kernel 20 to 26 percent slower than a
non-paged one. Microsoft's vAttention paper (2024) says you do not have to
accept that cost: you can keep the logical, virtual address space one
contiguous line, so reads are simple and fast, while still only committing
physical GPU memory in small blocks underneath it, using CUDA's virtual
memory system calls. Virtual addresses are just numbers you promise to use
later; physical memory is the actual chips being used. You can reserve a
huge range of virtual addresses for free, it costs you nothing until you
actually write real data into it, which is when physical memory gets
committed. vAttention exploits exactly that gap.

**Q: What did you actually build, and why no GPU?**

We do not have GPU access at the proposal stage, and vAttention's real
trick uses CUDA specific virtual memory calls that do not exist on CPU. So
we built a controlled behavioral analogy in pure PyTorch: three KV-cache
backends sharing one identical attention function, so cache management is
the only variable.

- `NaiveContiguousCache`: pre-allocates full max length up front. Fast
  reads, wastes memory. This is the pre-vLLM approach.
- `PagedCache`: allocates memory in 16 token blocks on demand, but has to
  gather and concatenate every block on every read. This is our
  reproduction of PagedAttention's actual mechanism, minus the CUDA kernel
  fusion.
- `VAttentionCache`: same on demand block accounting for memory, so the
  memory saving claim is preserved, but reads go through one contiguous
  buffer we extend by a single token each step, instead of re-gathering
  everything. This is the behavioral effect of vAttention's fix, not its
  actual CUDA mechanism; say that clearly if asked, do not overclaim it.

**Q: What did you measure, and what did you find?**

16 sequences, 4 layers, 8 heads, 64 dimensional heads, 256 decode steps,
identical random Q/K/V every step across all three backends, so timing
differences come only from cache management, not from what is being
computed.

- Paged: **332% slower** per decode step than the naive contiguous
  baseline.
- Our vAttention-style version: only **7.2% slower**, it recovers almost
  all of that gap.
- Physical memory: naive commits everything immediately from step 0;
  paged and vAttention-style grow identically, block by block, on demand,
  and their memory curves overlap exactly by construction.

**Q: Why is your 332% so much bigger than the paper's 20 to 26%?**

Because the paper compares two real, hand optimized fused CUDA kernels.
Our paged backend is a plain Python `for` loop over blocks calling
`torch.cat` every step. There is no kernel fusion, so the relative
overhead of doing several small operations versus doing one big operation
is naturally much larger in an unoptimized interpreted loop than in
compiled, fused GPU code. We say this directly in the proposal's
Limitations section instead of letting the number stand without context.
The direction of the result, paging costs real speed and decoupling
virtual from physical recovers it, is what we are claiming, not the exact
magnitude.

**Q: What is the block size trade-off, and why 16?**

Smaller blocks, say 4 tokens, mean less wasted memory at the tail end of a
sequence, less internal fragmentation, since the last block is rarely
full, but more blocks total means more block table entries to track and
more per read overhead. Bigger blocks mean less bookkeeping overhead but
more wasted memory per sequence. 16 is vLLM's own default, chosen
empirically as a reasonable middle point on real GPUs; we reused it so our
numbers are comparable to the real system rather than an arbitrary choice.

**Q: Why only the decode phase, not prefill?**

Prefill, processing the initial prompt, computes attention over a big
batch of tokens all at once, one large contiguous matrix operation
regardless of how the cache is laid out, because you are writing the
whole prompt's KV in one shot. Decode, generating one new token at a
time, is where the cache gets read and appended to repeatedly, one token
per step, which is exactly where block table lookups happen over and over
and the non-contiguous layout actually costs you something. That is why
both the vLLM ablation and our benchmark focus there.

**Q: What is AdaptiveBlockCache, and why did you add it?**

Mid-project feedback was that the first three backends just reproduce two
papers' own ideas, nothing in the approach was ours. Fair. So we built a
fourth backend that's actually our idea: `AdaptiveBlockCache`. Same
grow-a-list-of-blocks pattern as `PagedCache`, but blocks aren't all the
same size. Early blocks are small (16 to 8, half of vLLM's default),
because most real requests are short and small blocks keep the wasted
tail-end space tight for the common case. Once a sequence passes 4 times
the block size in tokens, later blocks switch to large (4x the default),
on the idea that fewer, bigger fragments means less to gather for the
sequences that do keep going. It's the same idea as size classes in a
real memory allocator (glibc's malloc) or young/old generations in a
garbage collector, just applied to KV-cache blocks. Neither the vLLM
paper nor the vAttention paper does this.

**Q: Did it work?**

Half of it did, half of it didn't, and the honest answer to "did it
work" is the actual finding, not a clean yes. It does reduce the
physical block count: about 11 blocks per sequence by the end of a
256-token run versus 16 for plain fixed-size paging, roughly 30 percent
fewer blocks, which is real, it's exactly what the design was supposed to
buy. But it did not translate into faster decoding. In our measurements,
mean step time was *higher* than plain paged, not lower, consistently,
even after we fixed a real bug where the implementation was redundantly
recomputing something inside a per-batch loop that only depends on
timestep, and even after rotating which backend runs first across trials
so machine load drift couldn't be quietly penalizing whichever one
happened to run last. We went further and isolated the gather step
itself with a controlled test, 2000 repetitions, properly warmed up, and
found that `torch.cat`'s cost did not meaningfully depend on how many
pieces or what sizes they were, which ruled out our own "fewer pieces
should be faster" assumption directly, instead of just accepting a
number that didn't match it.

**Q: So why report a backend that made things worse?**

Because it's a real, informative result, not a failure to hide. It tells
us something the clean reproductions alone don't: just reducing block
*count* is not enough to recover decode speed in a design that still
re-gathers everything from scratch every single step. That's exactly
why vAttention's actual fix isn't "use bigger blocks", it's "stop
re-gathering at all", true O(1) append through virtual contiguity. Our
own attempt at a cheaper mitigation makes vAttention's actual design
choice look more deliberate in hindsight, not less. That's the novelty
answer: not a bigger number, a genuine idea of our own, tested honestly,
with a result that sharpens the argument for why the real paper's fix
works the way it does.

**Q: Where is the honesty in this project, what would you not claim?**

- Not claiming to beat vLLM or vAttention's real numbers. This is a small
  scale mechanism check, not a production benchmark.
- Not claiming our physical memory saved number is measured RAM. It is an
  analytical block count, the same convention the papers themselves use
  to report memory savings, because Python cannot do real lazy physical
  page commit the way CUDA can.
- Not claiming vAttention-style is a CUDA virtual memory reproduction. It
  is a behavioral emulation of the same idea, clearly labeled as such.
- Not claiming any result about model output quality. Every experiment
  runs on random tensors, no real language model.
- Not claiming AdaptiveBlockCache is an improvement. It traded speed for
  a real memory win and we say so directly, instead of only reporting
  the half of the result that looks good.

**Q: What is next if this gets approved for full implementation?**

Get GPU access, Kaggle's free 30 hours per week or Colab, wire the same
four backends, or vAttention's real CUDA VMM implementation, into a
small real model like GPT-2 or TinyLlama, and re-run the same benchmark
methodology end to end with real prompts, ShareGPT style, the same source
vLLM's own paper used, to see if the same relative pattern, paging costs
speed and virtual physical decoupling recovers it, survives outside a CPU
microbenchmark.
