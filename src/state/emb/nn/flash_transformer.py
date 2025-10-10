# File: vci/flash_transformer.py
"""
This module implements a Transformer encoder layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_duplication_map(old_dim: int, new_dim: int):
    extra = new_dim - old_dim
    if extra <= 0:
        return []
    return [(i * old_dim) // extra for i in range(extra)]


class AdapterLayerNorm(nn.Module):
    def __init__(self, d_model: int, old_d_model: int):
        super().__init__()
        self.d_model = d_model
        self.old_d_model = old_d_model
        # Inner LayerNorm over the original subspace
        self.ln = nn.LayerNorm(old_d_model)

        # Cache duplication mapping for new dims → source old-dim index
        dup_map = _build_duplication_map(old_d_model, d_model)
        # store as buffer for device/dtype placement
        self.register_buffer("_dup_src_idx", torch.tensor(dup_map, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute LN statistics over only the original subspace
        x_old = x[..., : self.old_d_model]
        # Use the nn.LayerNorm module to normalize the old subspace (exact baseline behavior)
        norm_old = self.ln(x_old)

        # Reconstruct full-dim normalized output by tiling the normalized old features
        if self.d_model == self.old_d_model:
            return norm_old

        # Strict widening invariant: keep widened dims inert (zeros) at LN output.
        # This ensures zero tails pre-gating and prevents any accidental residual leakage.
        zeros_new = torch.zeros_like(x[..., self.old_d_model :])
        return torch.cat([norm_old, zeros_new], dim=-1)


class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, old_d_model: int | None = None):
        """
        Initializes the encoder layer.
        Args:
            d_model (int): model dimension.
            nhead (int): number of attention heads.
            dim_feedforward (int): dimension of the feed-forward network.
            dropout (float): dropout probability.
        """
        super().__init__()
        # Prefer deterministic math-based SDPA to minimize numeric drift across shapes
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout

        # Linear projections for Q, K, V in one matrix
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        # Residual gates to keep new dims off at initialization
        self.old_d_model = old_d_model if old_d_model is not None else d_model
        gate_init = torch.ones(d_model)
        if self.old_d_model < d_model:
            gate_init[self.old_d_model:] = 0.0
        self.register_buffer("gate_mask", gate_init)

        # LayerNorms (optionally adapter-wrapped to preserve old_d_model statistics)
        if old_d_model is not None and old_d_model < d_model:
            self.norm1 = AdapterLayerNorm(d_model, old_d_model)
            self.norm2 = AdapterLayerNorm(d_model, old_d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Tensor of shape (batch_size, seq_len, d_model)
            src_mask: (optional) attention mask.
            src_key_padding_mask: (optional) padding mask.
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # For this simple implementation, we'll use either one of the masks.
        # You can combine them as needed.
        mask = src_key_padding_mask if src_key_padding_mask is not None else src_mask

        # ----- Self-Attention Block -----
        residual = src

        # Compute Q, K, V projections in one go.
        qkv = self.qkv_proj(src)  # shape: (B, T, 3*d_model)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each: (B, T, d_model)

        # Reshape for multi-head attention.
        head_dim = self.d_model // self.nhead
        q = q.view(src.size(0), src.size(1), self.nhead, head_dim).transpose(1, 2)  # (B, nhead, T, head_dim)
        k = k.view(src.size(0), src.size(1), self.nhead, head_dim).transpose(1, 2)
        v = v.view(src.size(0), src.size(1), self.nhead, head_dim).transpose(1, 2)

        # Note: Avoid per-subspace Q/K amplitude tweaks; prior experiments degraded locality.

        # Use PyTorch’s built-in scaled_dot_product_attention.
        # Disable attention dropout in eval for deterministic behavior
        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=(self.dropout if self.training else 0.0), is_causal=False
        )
        # Merge heads.
        attn_output = attn_output.transpose(1, 2).contiguous().view(src.size(0), src.size(1), self.d_model)
        attn_output = self.out_proj(attn_output)
        # Gate new dimensions off at init to preserve residual behavior
        attn_output = attn_output * self.gate_mask
        src = self.norm1(residual + self.dropout_layer(attn_output))
        # Keep widened dims strictly zero on the residual stream at init
        src = src * self.gate_mask

        # ----- Feed-Forward Block -----
        residual2 = src
        ff_output = self.linear2(self.dropout_layer(F.gelu(self.linear1(src))))
        # Gate new dimensions off at init to preserve residual behavior
        ff_output = ff_output * self.gate_mask
        src = self.norm2(residual2 + self.dropout_layer(ff_output))
        # Keep widened dims strictly zero on the residual stream at init
        src = src * self.gate_mask
        return src


class FlashTransformerEncoder(nn.Module):
    def __init__(self, layers):
        """
        A simple encoder that applies a stack of FlashTransformerEncoderLayer instances.
        Args:
            layers (list[nn.Module]): list of FlashTransformerEncoderLayer instances.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Applies each encoder layer in sequence.
        Args:
            src: Tensor of shape (B, T, d_model)
            src_mask: (optional) attention mask.
            src_key_padding_mask: (optional) padding mask.
        Returns:
            Tensor of shape (B, T, d_model)
        """
        # Use src_key_padding_mask if provided; otherwise use src_mask.
        mask = src_key_padding_mask if src_key_padding_mask is not None else src_mask
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=mask)
        return output
