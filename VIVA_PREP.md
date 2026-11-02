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
fourth backend that's actually our idea: `AdaptiveBlockCache`. It is a
single growing buffer per layer, like `VAttentionCache`, except nothing
is preallocated to `max_seq_len` up front. It grows lazily in chunks, the
same way a dynamic array (Python list, C++ vector) doubles when it runs
out of room: small chunks early (half of vLLM's default block size),
because most real requests are short and this keeps wasted tail-end
space tight for the common case, bigger chunks once a sequence survives
past 4 times the block size in tokens (4x the default), because fewer,
bigger reallocations are cheaper for the sequences that do keep going.
It's the same idea as size classes in a real memory allocator (glibc's
malloc) or young/old generations in a garbage collector, just applied to
KV-cache growth. Neither the vLLM paper nor the vAttention paper does
this.

**Q: Did it work?**

Not on the first attempt, and that first attempt is worth being honest
about because it's exactly what the live comprehension round will probe.
Our first version kept `PagedCache`'s per-sequence Python-list-of-blocks
structure, just with uneven block sizes. It did cut the block count by
about 30 percent, but it measured *slower* than plain fixed-size paging,
consistently, even after we fixed a real bug (redundant per-batch
recomputation) and controlled for run-order bias (rotating which backend
runs first across trials). We did not accept that number and move on; we
dug into why. A controlled, warmed-up `torch.cat` microbenchmark ruled
out our first guess, that fewer pieces should mean a cheaper gather, the
cost barely moved with piece count. The real cause was structural, not
about blocks at all: `PagedCache` (and our first attempt) store each
sequence in a separate Python list, so producing the one tensor the
attention math needs forces a real memory copy on every single decode
step, regardless of block size. `VAttentionCache` and `NaiveContiguousCache`
never pay that cost because they store the whole batch in one tensor
from the start, so a read is a free slice, not a copy. We isolated this
directly: stacking 16 separate per-sequence tensors every step over a
256-step run cost about 30 times more than slicing one pre-shaped tensor
the same number of times, same data, same step count. That is the actual
bottleneck, not block count, and it explains why neither our first
attempt nor plain `PagedCache` ever had a chance of beating
`VAttentionCache`, no matter how the blocks were sized.

Once we rebuilt `AdaptiveBlockCache` on that same single-tensor structure
as `VAttentionCache`, with the buffer grown lazily in adaptive-sized
chunks instead of preallocated up front, it landed in the same speed
class as naive and vAttention-style, well within normal CPU noise,
while still keeping the real memory win: about 30 percent fewer blocks
than either, because its chunk size adapts to how long a sequence has
actually run instead of committing a fixed size every time.

**Q: So what is the actual result and why does it matter?**

`AdaptiveBlockCache` is now fast (same class as naive and vAttention-style,
tens of a percent, not hundreds like paged) and uses fewer blocks than
every other backend, including vAttention-style's fixed granularity. The
reason this is a real idea and not just a copy of vAttention: vAttention's
own fix is to preallocate one contiguous buffer per sequence up front and
never re-gather. That works, but it means committing `max_seq_len` worth
of memory immediately even for a sequence that ends after ten tokens.
`AdaptiveBlockCache` keeps vAttention's actual read-path mechanism, we are
not claiming a new read mechanism, reusing a proven one deliberately is
the right call, but changes how physical memory gets committed underneath
it: lazily, in adaptively-sized chunks, so short sequences (the common
case in real traffic) never over-commit. The debugging story itself, ruling
out a wrong hypothesis with a controlled test instead of accepting a bad
number, is also a legitimate answer to "what did you actually do here",
not a weakness to hide.

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
- Not claiming AdaptiveBlockCache's read mechanism is novel. It is the
  same single-tensor, no-re-gather mechanism as VAttentionCache, reused
  on purpose; our contribution is the lazy, adaptively-sized growth
  policy underneath it, and we say exactly which part is ours and which
  part is reused, instead of blurring the two together.
- Not hiding that our first version of AdaptiveBlockCache (PagedCache's
  block-list structure with uneven block sizes) did not work, it measured
  slower than plain paging despite using fewer blocks. We report why it
  failed and what we changed, not just the final working version.

**Q: What is next if this gets approved for full implementation?**

Get GPU access, Kaggle's free 30 hours per week or Colab, wire the same
four backends, or vAttention's real CUDA VMM implementation, into a
small real model like GPT-2 or TinyLlama, and re-run the same benchmark
methodology end to end with real prompts, ShareGPT style, the same source
vLLM's own paper used, to see if the same relative pattern, paging costs
speed and virtual physical decoupling recovers it, survives outside a CPU
microbenchmark.
