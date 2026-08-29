"""
Shared multi-head attention math used by every KV-cache strategy in this
project. This file never changes across experiments -- the only thing that
differs between the naive, paged, and vattention-style caches (see caches.py)
is HOW they store/fetch K and V. That keeps the benchmark fair: any speed
difference we measure comes from memory management, not from the attention
math itself. This mirrors how the vLLM and vAttention papers isolate the
KV-cache management cost from the attention kernel cost.
"""

import math
import torch


def decode_step_attention(q, k_hist, v_hist):
    """
    One autoregressive decode step of causal multi-head self-attention.

    q:       [batch, n_heads, 1, head_dim]        - query for the new token
    k_hist:  [batch, n_heads, seq_len, head_dim]   - all keys seen so far
    v_hist:  [batch, n_heads, seq_len, head_dim]   - all values seen so far

    Returns: [batch, n_heads, 1, head_dim] - attention output for the new token
    """
    head_dim = q.shape[-1]
    scores = torch.matmul(q, k_hist.transpose(-2, -1)) / math.sqrt(head_dim)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_hist)
    return out
