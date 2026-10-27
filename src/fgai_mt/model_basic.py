"""Encoder-decoder Transformer implemented from scratch (Vaswani et al., 2017), with KV-cache decoding."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, L, D]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return self.dropout(x + self.pe[:, offset : offset + x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.h = d_model, n_heads
        self.d_k = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.d_k).transpose(1, 2)

    def forward(self, x, kv=None, mask=None, past=None):
        """x: queries [B,T,D]; kv: keys/values source [B,S,D] (None -> self-attention)."""
        q = self._split(self.wq(x))
        kv_input = x if kv is None else kv
        k_new = self._split(self.wk(kv_input))
        v_new = self._split(self.wv(kv_input))
        if past is not None:
            k = torch.cat([past[0], k_new], dim=2)
            v = torch.cat([past[1], v_new], dim=2)
        else:
            k, v = k_new, v_new
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        return self.out(out), (k, v)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        h, _ = self.self_attn(self.ln1(x), mask=src_mask)
        x = x + self.drop(h)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, past=None):
        h, present = self.self_attn(self.ln1(x), mask=tgt_mask, past=past)
        x = x + self.drop(h)
        h, _ = self.cross_attn(self.ln2(x), kv=memory, mask=memory_mask)
        x = x + self.drop(h)
        x = x + self.ff(self.ln3(x))
        return x, present


class TransformerMT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
                 d_ff=1024, dropout=0.1, pad_id=0, max_len=512):
        super().__init__()
        self.pad_id, self.d_model = pad_id, d_model
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.src_pe = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pe = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_decoder_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.ln_f_dec = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tgt_embed.weight  # weight tying
        self.scale = math.sqrt(d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small init keeps the initial cross-entropy near ln(V) instead of exploding."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # -- encoder / decoder -------------------------------------------------
    def encode(self, src, src_key_padding_mask=None):
        x = self.src_pe(self.src_embed(src) * self.scale)
        mask = None
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask[:, None, None, :]
        for layer in self.encoder:
            x = layer(x, mask)
        return self.ln_f(x)

    def decode(self, tgt_in, memory, tgt_pad_mask=None, memory_mask=None, past=None):
        x = self.tgt_pe(self.tgt_embed(tgt_in) * self.scale)
        t, s = tgt_in.size(1), memory.size(1)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tgt_in.device), diagonal=1)[None, None]
        mask = causal
        if tgt_pad_mask is not None:
            mask = mask | tgt_pad_mask[:, None, None, :]
        caches = []
        for i, layer in enumerate(self.decoder):
            p = past[i] if past is not None else None
            x, present = layer(x, memory, tgt_mask=mask, memory_mask=memory_mask, past=p)
            caches.append(present)
        logits = self.lm_head(self.ln_f_dec(x))
        return logits, caches

    def forward(self, src, tgt_in):
        src_pad = src.eq(self.pad_id)
        tgt_pad = tgt_in.eq(self.pad_id)
        memory = self.encode(src, src_pad)
        memory_mask = src_pad[:, None, None, :]
        logits, _ = self.decode(tgt_in, memory, tgt_pad, memory_mask)
        return logits

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def decode_step(self, tokens, memory, memory_mask, past=None):
        x = self.tgt_pe(self.tgt_embed(tokens) * self.scale, offset=past[0][0].size(2) if past is not None else 0)
        caches = []
        for i, layer in enumerate(self.decoder):
            x, present = layer(x, memory, tgt_mask=None, memory_mask=memory_mask,
                               past=past[i] if past is not None else None)
            caches.append(present)
        return self.lm_head(self.ln_f_dec(x))[:, -1, :], caches


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


@torch.no_grad()
def greedy_decode(model: TransformerMT, src, max_len=100, bos_id=1, eos_id=2, pad_id=0, device=None):
    device = device or src.device
    src_pad = src.eq(pad_id)
    memory = model.encode(src, src_pad)
    memory_mask = src_pad[:, None, None, :]
    b = src.size(0)
    tokens = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(b, dtype=torch.bool, device=device)
    caches = None
    for _ in range(max_len):
        logits, caches = model.decode_step(tokens[:, -1:], memory, memory_mask, caches)
        nxt = logits.argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
        tokens = torch.cat([tokens, nxt[:, None]], dim=1)
        finished |= nxt.eq(eos_id)
        if bool(finished.all()):
            break
    return tokens[:, 1:]


@torch.no_grad()
def beam_search(model: TransformerMT, src, beam_size=4, max_len=100, bos_id=1, eos_id=2, pad_id=0,
                length_penalty=0.8, device=None):
    """Batched beam search with KV cache; returns token ids per sentence (without BOS)."""
    device = device or src.device
    b = src.size(0)
    src_pad = src.eq(pad_id)
    memory = model.encode(src, src_pad)
    mem_mask = src_pad[:, None, None, :]
    if beam_size > 1:
        memory = memory.repeat_interleave(beam_size, dim=0)
        mem_mask = mem_mask.repeat_interleave(beam_size, dim=0)
    n = b * beam_size

    tokens = torch.full((n, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.full((n,), float("-inf"), device=device)
    scores.view(b, beam_size)[:, 0] = 0.0  # only the first beam is alive at step 0
    lengths = torch.ones(n, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    caches = None

    for step in range(max_len):
        logits, caches = model.decode_step(tokens[:, -1:], memory, mem_mask, caches)
        logp = F.log_softmax(logits.float(), dim=-1)
        if step == 0:
            logp[:, eos_id] = float("-inf")
        logp[finished] = float("-inf")
        logp[finished, eos_id] = 0.0

        cand = (scores[:, None] + logp).view(b, beam_size * logp.size(-1))
        top_scores, top_idx = cand.topk(beam_size, dim=-1)
        beam_idx = (top_idx // logp.size(-1)).view(-1)
        tok_idx = (top_idx % logp.size(-1)).view(-1)

        # reorder caches and running state
        index = (torch.arange(b, device=device)[:, None] * beam_size + beam_idx.view(b, beam_size)).view(-1)
        if caches is not None:
            caches = [(k.index_select(0, index), v.index_select(0, index)) for k, v in caches]
        was_finished = finished.index_select(0, index)
        finished = was_finished
        lengths = lengths.index_select(0, index) + (~was_finished).float()
        scores = top_scores.view(-1)
        tokens = torch.cat([tokens.index_select(0, index), tok_idx[:, None]], dim=1)
        finished = finished | tok_idx.eq(eos_id)

        if bool(finished.all()):
            break

    # rank hypotheses with length normalisation
    out = []
    scores = scores.view(b, beam_size)
    lengths = lengths.view(b, beam_size)
    norm = scores / lengths.clamp(min=1).pow(length_penalty)
    best = norm.argmax(dim=-1)
    for i in range(b):
        j = i * beam_size + int(best[i])
        seq = tokens[j, 1:].tolist()
        if eos_id in seq:
            seq = seq[: seq.index(eos_id)]
        out.append(seq)
    return out
