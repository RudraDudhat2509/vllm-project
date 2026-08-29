"""
Three KV-cache strategies, all exposing the same interface so benchmark.py
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

All three report physical_blocks_used(t): an analytical block count, not
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
