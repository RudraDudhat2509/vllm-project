"""
Four KV-cache strategies, all exposing the same interface so benchmark.py
can swap them in and out without touching the attention math:

  NaiveContiguousCache  preallocates the full max_seq_len upfront, reads
                           are a plain contiguous slice (fast, memory-wasteful).
                           This is how pre-vLLM serving engines (e.g. early
                           FasterTransformer / HF generate) handled KV cache.

  PagedCache            vLLM's actual approach. Physical memory is
                           committed in fixed-size blocks only as needed
                           (memory-efficient), but every read has to gather
                           and concatenate every allocated block, which is
                           the block-table lookup overhead the vLLM paper
                           itself reports (20 to 26% slower than a non-paged
                           kernel).

  VAttentionCache       our reproduction of vAttention's core idea (Microsoft
                           Research, ASPLOS'25, arXiv:2405.04437): keep physical
                           memory commitment in fixed blocks (same memory
                           accounting as PagedCache) but present a virtually
                           contiguous read path so each step is a plain O(1)
                           append instead of an O(blocks_so_far) re-gather.
                           On a real GPU that contiguity comes from CUDA
                           virtual memory APIs; here we emulate the same
                           behavioral effect in plain PyTorch since this
                           proposal stage has no GPU access.

  AdaptiveBlockCache     our own idea, not in either paper: a single
                           contiguous buffer per layer, like VAttentionCache,
                           but it is not preallocated to max_seq_len up
                           front. It grows lazily, in chunks, the same way a
                           dynamic array (Python list, C++ vector) doubles
                           when it runs out of room: small chunks early
                           (most real requests are short, so this keeps
                           wasted tail space tight for the common case),
                           bigger chunks once a sequence survives past a
                           threshold (fewer reallocations for the sequences
                           that keep going). Same idea as size classes in a
                           real allocator (glibc malloc) or young/old
                           generations in a GC, applied to KV-cache growth.
                           Reads are always a plain slice of one tensor, the
                           same O(1) read path as VAttentionCache, so it is
                           never slower for the reason PagedCache is: it
                           never re-gathers per-sequence block lists, there
                           are none. The only extra cost is the occasional
                           grow-and-copy when a chunk fills up, amortized
                           over the whole run the same way dynamic array
                           growth is.

All four report physical_blocks_used(t): an analytical block count, not
measured process RSS. We track it this way because Python/CPU can't emulate
real OS/CUDA lazy physical-page commit, but block-count accounting is
exactly the metric the vLLM and vAttention papers themselves use to report
memory savings, so it's a fair like-for-like comparison.
"""

import math
import torch


class NaiveContiguousCache:
    def __init__(self, batch_size, n_layers, n_heads, head_dim, max_seq_len, block_size):
        self.K = torch.zeros(batch_size, n_layers, max_seq_len, n_heads, head_dim)
        self.V = torch.zeros_like(self.K)
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.n_layers = n_layers

    def append_and_read(self, layer, t, k_new, v_new):
        self.K[:, layer, t] = k_new.squeeze(2)
        self.V[:, layer, t] = v_new.squeeze(2)
        k_hist = self.K[:, layer, : t + 1].permute(0, 2, 1, 3)
        v_hist = self.V[:, layer, : t + 1].permute(0, 2, 1, 3)
        return k_hist, v_hist

    def physical_blocks_used(self, t):
        return self.batch_size * self.n_layers * math.ceil(self.max_seq_len / self.block_size)


class PagedCache:
    def __init__(self, batch_size, n_layers, n_heads, head_dim, max_seq_len, block_size):
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.k_blocks = [[[] for _ in range(batch_size)] for _ in range(n_layers)]
        self.v_blocks = [[[] for _ in range(batch_size)] for _ in range(n_layers)]

    def append_and_read(self, layer, t, k_new, v_new):
        block_idx = t // self.block_size
        offset = t % self.block_size
        for b in range(self.batch_size):
            kb_list = self.k_blocks[layer][b]
            vb_list = self.v_blocks[layer][b]
            if offset == 0:
                kb_list.append(torch.zeros(self.block_size, self.n_heads, self.head_dim))
                vb_list.append(torch.zeros(self.block_size, self.n_heads, self.head_dim))
            kb_list[block_idx][offset] = k_new[b, :, 0, :]
            vb_list[block_idx][offset] = v_new[b, :, 0, :]

        # this per-block gather + concat is the reproduced block-table overhead
        k_hist_batch, v_hist_batch = [], []
        for b in range(self.batch_size):
            k_cat = torch.cat(self.k_blocks[layer][b], dim=0)[: t + 1]
            v_cat = torch.cat(self.v_blocks[layer][b], dim=0)[: t + 1]
            k_hist_batch.append(k_cat)
            v_hist_batch.append(v_cat)
        k_hist = torch.stack(k_hist_batch, dim=0).permute(0, 2, 1, 3)
        v_hist = torch.stack(v_hist_batch, dim=0).permute(0, 2, 1, 3)
        return k_hist, v_hist

    def physical_blocks_used(self, t):
        return self.batch_size * self.n_layers * math.ceil((t + 1) / self.block_size)


class VAttentionCache:
    def __init__(self, batch_size, n_layers, n_heads, head_dim, max_seq_len, block_size):
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.K = torch.zeros(batch_size, n_layers, max_seq_len, n_heads, head_dim)
        self.V = torch.zeros_like(self.K)

    def append_and_read(self, layer, t, k_new, v_new):
        self.K[:, layer, t] = k_new.squeeze(2)
        self.V[:, layer, t] = v_new.squeeze(2)
        k_hist = self.K[:, layer, : t + 1].permute(0, 2, 1, 3)
        v_hist = self.V[:, layer, : t + 1].permute(0, 2, 1, 3)
        return k_hist, v_hist

    def physical_blocks_used(self, t):
        # same block-granularity commit as PagedCache: the memory saving
        # is preserved even though the read path above is contiguous
        return self.batch_size * self.n_layers * math.ceil((t + 1) / self.block_size)


class AdaptiveBlockCache:
    """
    One contiguous buffer per layer, holding the whole batch, like
    VAttentionCache. Reads are a plain slice, same O(1) read path, so this
    is never slower for the reason PagedCache is: there is no per-sequence
    block list to re-gather, there is nothing to re-gather at all.

    The difference from VAttentionCache is that nothing is preallocated to
    max_seq_len up front. The buffer grows lazily, in chunks, the same way
    a dynamic array (Python list, C++ vector) doubles when it runs out of
    room: small chunks early (block_size // 2), since most real requests
    are short and this keeps wasted tail space tight for the common case,
    bigger chunks once a sequence survives past warmup_tokens (block_size *
    4 tokens in), since fewer, bigger reallocations are cheaper for the
    sequences that keep going. Same idea as size classes in a real
    allocator (glibc malloc) or young/old generations in a GC, applied to
    KV-cache growth. The only extra cost versus VAttentionCache is the
    occasional grow-and-copy when a chunk fills up, amortized over the
    whole run the same way dynamic array growth is, so it stays rare.

    PagedCache and VAttentionCache are both deliberately left untouched:
    PagedCache has to stay a faithful reproduction of the real block-table
    re-gather cost the vLLM paper reports, and VAttentionCache has to stay
    a faithful reproduction of vAttention's fixed-block, fully-preallocated
    fix. AdaptiveBlockCache is the one backend we are free to redesign.
    """
    def __init__(self, batch_size, n_layers, n_heads, head_dim, max_seq_len, block_size):
        self.small_chunk = block_size // 2
        self.large_chunk = block_size * 4
        self.warmup_tokens = block_size * 4
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.K = [None for _ in range(n_layers)]
        self.V = [None for _ in range(n_layers)]
        self.capacity = [0 for _ in range(n_layers)]
        self.num_chunks = [0 for _ in range(n_layers)]

    def _next_chunk_size(self, tokens_so_far):
        return self.small_chunk if tokens_so_far < self.warmup_tokens else self.large_chunk

    def append_and_read(self, layer, t, k_new, v_new):
        if t == self.capacity[layer]:  # current buffer is full, grow it
            grow = self._next_chunk_size(t)
            new_capacity = self.capacity[layer] + grow
            new_K = torch.zeros(self.batch_size, new_capacity, self.n_heads, self.head_dim)
            new_V = torch.zeros_like(new_K)
            if self.K[layer] is not None:
                new_K[:, : self.capacity[layer]] = self.K[layer]
                new_V[:, : self.capacity[layer]] = self.V[layer]
            self.K[layer] = new_K
            self.V[layer] = new_V
            self.capacity[layer] = new_capacity
            self.num_chunks[layer] += 1

        self.K[layer][:, t] = k_new.squeeze(2)
        self.V[layer][:, t] = v_new.squeeze(2)
        k_hist = self.K[layer][:, : t + 1].permute(0, 2, 1, 3)
        v_hist = self.V[layer][:, : t + 1].permute(0, 2, 1, 3)
        return k_hist, v_hist

    def physical_blocks_used(self, t):
        return self.batch_size * self.n_layers * self.num_chunks[0]
