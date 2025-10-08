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
        self.down = nn.Linear(d_model, old_d_model, bias=False)
        self.ln = nn.LayerNorm(old_d_model)
        self.up = nn.Linear(old_d_model, d_model, bias=False)

        # Initialize down as selector of first old_d_model features
        with torch.no_grad():
            self.down.weight.zero_()
            for i in range(old_d_model):
                self.down.weight[i, i] = 1.0

            # Initialize up as identity on first old_d_model and duplicate for extra dims
            self.up.weight.zero_()
            for i in range(old_d_model):
                self.up.weight[i, i] = 1.0

            dup_map = _build_duplication_map(old_d_model, d_model)
            for idx, src in enumerate(dup_map):
                self.up.weight[old_d_model + idx, src] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        y = self.ln(y)
        z = self.up(y)
        return z


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
        torch.backends.cuda.enable_flash_sdp(True)

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout

        # Linear projections for Q, K, V in one matrix
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

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

        # Use PyTorch’s built-in scaled_dot_product_attention.
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=False)
        # Merge heads.
        attn_output = attn_output.transpose(1, 2).contiguous().view(src.size(0), src.size(1), self.d_model)
        attn_output = self.out_proj(attn_output)
        src = self.norm1(residual + self.dropout_layer(attn_output))

        # ----- Feed-Forward Block -----
        residual2 = src
        ff_output = self.linear2(self.dropout_layer(F.gelu(self.linear1(src))))
        src = self.norm2(residual2 + self.dropout_layer(ff_output))
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
