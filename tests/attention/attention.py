"""
Transformer minimal-yet-robust implementation (PyTorch)
- Scaled Dot-Product Attention
- Multi-Head Attention (self & cross)
- Position-wise FFN
- Sinusoidal Positional Encoding
- Encoder/Decoder layers & stacks
- Full Encoder-Decoder model with embedding + generator
- Greedy decoding with causal/padding masking

Design goals:
1) Readable: mirrors equations; minimal magic.
2) Hackable: easy to swap with your own backend (e.g., np or custom kernels).
3) Correct masking: padding and autoregressive masks both supported.

Tested on PyTorch >= 2.1.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities: masks & helpers
# -----------------------------
def subsequent_mask(sz: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return a (sz, sz) additive mask with -inf above the diagonal (prevent attending to future).
    Shapes follow (T, S) semantics. Add to attention scores.
    """
    mask = torch.full((sz, sz), float('-inf'), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.masked_fill(mask == 0, 0.0)  # lower triangle inc. diag -> 0
    return mask  # (T, S)


def make_pad_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create a boolean padding mask (B, S) where True indicates PAD positions.
    - lengths: (B,) lengths of each sequence (non-padded tokens count)
    - max_len: optional override; otherwise use lengths.max()
    """
    B = lengths.size(0)
    S = int(max_len or lengths.max().item())
    arange = torch.arange(S, device=lengths.device)
    mask = arange.unsqueeze(0).expand(B, S) >= lengths.unsqueeze(1)
    return mask  # (B, S) True = PAD


# -----------------------------
# Core: Scaled Dot-Product Attention
# -----------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, H, T, Dh)
        k: torch.Tensor,  # (B, H, S, Dh)
        v: torch.Tensor,  # (B, H, S, Dh)
        attn_bias: Optional[torch.Tensor] = None,  # (T, S) or (B, 1, T, S) additive
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, S) bool: True=PAD
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Dh = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,T,S)

        if attn_bias is not None:
            # Broadcast (T,S) -> (1,1,T,S) or accept (B,1,T,S)
            if attn_bias.dim() == 2:
                attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
            scores = scores + attn_bias

        if key_padding_mask is not None:
            # Convert (B,S) bool to additive mask (B,1,1,S)
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B,H,T,Dh)
        return out, attn


# -----------------------------
# Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.attn = ScaledDotProductAttention(dropout)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.num_heads
        x = x.view(B, T, H, self.d_head).transpose(1, 2)  # (B,H,T,Dh)
        return x

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous().view(B, T, H * Dh)  # (B,T,C)
        return x

    def forward(
        self,
        x_q: torch.Tensor,             # (B, Tq, C)
        x_kv: Optional[torch.Tensor] = None,  # (B, Tk, C) for cross-attn; None -> self-attn
        attn_bias: Optional[torch.Tensor] = None,  # (Tq, Tk) or (B,1,Tq,Tk)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, Tk)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_kv is None:
            x_kv = x_q

        q = self.split_heads(self.w_q(x_q))  # (B,H,Tq,Dh)
        k = self.split_heads(self.w_k(x_kv))  # (B,H,Tk,Dh)
        v = self.split_heads(self.w_v(x_kv))  # (B,H,Tk,Dh)

        out, attn = self.attn(q, k, v, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        out = self.merge_heads(out)
        out = self.proj(out)
        out = self.dropout(out)
        return out, attn


# -----------------------------
# Position-wise FeedForward
# -----------------------------
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Positional Encoding (sinusoidal)
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, start_pos:start_pos + T]


# -----------------------------
# Encoder/Decoder layers
# -----------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (no causal mask in encoder)
        attn_out, _ = self.self_attn(x, x_kv=None, attn_bias=None, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,                 # (B, T, C)
        memory: torch.Tensor,            # (B, S, C)
        self_attn_bias: torch.Tensor,    # (T, T) causal
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T)
        memory_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S)
    ) -> torch.Tensor:
        sa_out, _ = self.self_attn(x, attn_bias=self_attn_bias, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + sa_out)

        ca_out, _ = self.cross_attn(x, x_kv=memory, attn_bias=None, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + ca_out)

        x = self.norm3(x + self.ffn(x))
        return x


# -----------------------------
# Stacks
# -----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer if i == 0 else type(layer)(
            layer.self_attn.d_model,
            layer.self_attn.num_heads,
            layer.ffn.net[0].out_features,
            layer.ffn.net[2].p,  # dropout
        ) for i in range(num_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for l in self.layers:
            x = l(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer if i == 0 else type(layer)(
            layer.self_attn.d_model,
            layer.self_attn.num_heads,
            layer.ffn.net[0].out_features,
            layer.ffn.net[2].p,
        ) for i in range(num_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_bias: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for l in self.layers:
            x = l(
                x,
                memory,
                self_attn_bias=self_attn_bias,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(x)


# -----------------------------
# Full Model
# -----------------------------
@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout: float = 0.1
    max_len: int = 2048
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.src_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.tgt_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_len)

        enc_layer = TransformerEncoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout)
        dec_layer = TransformerDecoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout)
        self.encoder = TransformerEncoder(enc_layer, cfg.num_encoder_layers)
        self.decoder = TransformerDecoder(dec_layer, cfg.num_decoder_layers)

        self.generator = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.dropout = nn.Dropout(cfg.dropout)

    # ---- Encoding/Decoding helpers ----
    def encode(self, src_ids: torch.Tensor, src_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # src_ids: (B, S)
        src_mask = make_pad_mask(src_lengths, max_len=src_ids.size(1))  # (B, S)
        x = self.src_embed(src_ids) * math.sqrt(self.cfg.d_model)
        x = self.pos_enc(self.dropout(x))
        memory = self.encoder(x, key_padding_mask=src_mask)
        return memory, src_mask

    def decode(self, tgt_ids: torch.Tensor, memory: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # tgt_ids: (B, T)
        T = tgt_ids.size(1)
        tgt_lengths = (tgt_ids != self.cfg.pad_id).sum(dim=1)
        tgt_pad_mask = make_pad_mask(tgt_lengths, max_len=T)  # (B, T)
        causal_bias = subsequent_mask(T, device=tgt_ids.device, dtype=memory.dtype)  # (T, T)

        x = self.tgt_embed(tgt_ids) * math.sqrt(self.cfg.d_model)
        x = self.pos_enc(self.dropout(x))
        hs = self.decoder(
            x,
            memory,
            self_attn_bias=causal_bias,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.generator(hs)  # logits (B, T, V)

    # ---- Full forward (teacher forcing style) ----
    def forward(self, src_ids: torch.Tensor, src_lengths: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, src_mask = self.encode(src_ids, src_lengths)
        logits = self.decode(tgt_in, memory, src_key_padding_mask=src_mask)
        return logits

    # ---- Greedy decode for inference ----
    @torch.no_grad()
    def generate(self, src_ids: torch.Tensor, src_lengths: torch.Tensor, max_new_tokens: int = 64) -> torch.Tensor:
        device = src_ids.device
        memory, src_mask = self.encode(src_ids, src_lengths)
        B = src_ids.size(0)
        ys = torch.full((B, 1), self.cfg.bos_id, dtype=torch.long, device=device)
        ended = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decode(ys, memory, src_key_padding_mask=src_mask)  # (B, T, V)
            next_token = logits[:, -1, :].argmax(dim=-1)  # greedy
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            ended = ended | (next_token == self.cfg.eos_id)
            if ended.all():
                break
        return ys


# -----------------------------
# Smoke tests
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=1000, d_model=128, num_heads=8, d_ff=256,
                            num_encoder_layers=2, num_decoder_layers=2, dropout=0.1,
                            max_len=256, pad_id=0, bos_id=1, eos_id=2)
    model = Transformer(cfg)

    B, S, T = 2, 11, 9
    src_lengths = torch.tensor([11, 7])
    src = torch.randint(3, cfg.vocab_size, (B, S))
    # pad second sample tail
    src[1, 7:] = cfg.pad_id

    tgt_in = torch.randint(3, cfg.vocab_size, (B, T))
    tgt_in[1, 6:] = cfg.pad_id

    logits = model(src, src_lengths, tgt_in)
    print("logits:", logits.shape)  # (B,T,V)

    # Greedy generate
    out = model.generate(src, src_lengths, max_new_tokens=5)
    print("generated ids:", out)

"""
Notes on porting to your custom AI framework
--------------------------------------------
- Replace nn.Linear with your Dense layer; preserve shapes.
- Replace matmul/softmax with your backend kernels. Key ops:
  scores = (q @ k^T) / sqrt(Dh); attn = softmax(scores, axis=-1); out = attn @ v
- Masks: add `attn_bias` (float with 0 or -inf) and use boolean key_padding_mask to set -inf before softmax.
- Sinusoidal positional encoding is pure tensor ops; can precompute once up to max_len.
- For inference caching, extend MultiHeadAttention to accept precomputed K/V and concat along sequence dim.
"""
