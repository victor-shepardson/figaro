"""Seq2SeqModule forward pass (for training) with FlexAttention, using the weights of its
HF Bert encoder/decoder.

relative_key_query scores (q.k + q.r(i-j) + k.r(i-j)) are computed with a score_mod
that looks up precomputed q.r and k.r tables of shape (b, h, L, n_distances), with the
distance i-j clipped to [-max_distance, max_distance], so memory is linear in the
sequence length rather than quadratic. Up to max_distance (255 by default, the largest
distance seen in training with context_size 256) this matches the HF attention exactly.

Optionally the decoder self-attention is restricted to a sliding window of the last
`window` tokens (cross-attention and the encoder stay full).

Differences from Seq2SeqModule.forward in training mode: no dropout on the attention
probabilities (FlexAttention doesn't support it); other dropout is as in HF Bert.

FlexAttention backward needs CUDA (torch 2.14 raises NotImplementedError for backward
on CPU and MPS); the forward pass runs anywhere. impl='account' replaces the attention
with a stand-in that returns meaningless values but saves the same tensors for backward
as flex_attention does, for estimating training memory on any device.
"""
import math

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from .fast import _merge_heads, _split_heads

_compiled_flex_attention = None


def _flex_attention(*args, compiled=True, **kwargs):
  global _compiled_flex_attention
  if not compiled:
    return flex_attention(*args, **kwargs)
  if _compiled_flex_attention is None:
    _compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
  return _compiled_flex_attention(*args, **kwargs)


class _AccountingAttention(torch.autograd.Function):
  """Stand-in for flex_attention that saves what it saves for backward (q, k, v, the
  output, the logsumexp and the captured score_mod tables) without computing anything."""
  @staticmethod
  def forward(ctx, q, k, v, q_table, k_table):
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
    ctx.save_for_backward(q, k, v, out, lse, q_table, k_table)
    return out

  @staticmethod
  def backward(ctx, grad):
    raise NotImplementedError('impl="account" is only for measuring saved tensors')


_block_masks = {}


def _block_mask(L, R, causal, window, device):
  if not causal:
    return None
  key = (L, R, window, str(device))
  if key not in _block_masks:
    if window is None:
      def mask_mod(b, h, i, j):
        return i >= j
    else:
      def mask_mod(b, h, i, j):
        return (i >= j) & (i - j < window)
    _block_masks[key] = create_block_mask(mask_mod, None, None, L, R, device=device)
  return _block_masks[key]


def relative_attention(q, k, v, distance_embedding, causal=False, max_distance=255,
                       window=None, impl='compiled'):
  """relative_key_query attention with clipped distances.

  q: (b, h, L, d) at positions 0..L-1; k, v: (b, h, R, d) at positions 0..R-1
  window: if causal, attend only to the last `window` positions (including the query's)
  impl: 'compiled' (torch.compile'd flex_attention), 'eager' or 'account'
  """
  b, h, L, d = q.shape
  R = k.shape[2]
  # range of distances i - j that occur, after clipping
  d_hi = min(max_distance, L - 1)
  if causal and window is not None:
    d_hi = min(d_hi, window - 1)
  d_lo = 0 if causal else max(-max_distance, -(R - 1))
  max_pos = (distance_embedding.num_embeddings + 1) // 2
  table = distance_embedding.weight[d_lo + max_pos - 1 : d_hi + max_pos]
  # flex_attention scales q.k by 1/sqrt(d) before score_mod; HF scales the sum
  scale = 1 / math.sqrt(d)
  q_table = (q @ table.T) * scale
  k_table = (k @ table.T) * scale
  if impl == 'account':
    return _AccountingAttention.apply(q, k, v, q_table, k_table)

  def score_mod(score, b, h, i, j):
    index = torch.clamp(i - j, d_lo, d_hi) - d_lo
    return score + q_table[b, h, i, index] + k_table[b, h, j, index]

  return _flex_attention(
    q, k, v, score_mod=score_mod, block_mask=_block_mask(L, R, causal, window, q.device),
    compiled=impl == 'compiled')


def _attention_block(attention, hidden, kv_hidden=None, **kwargs):
  """A BertAttention module (self-attention, or cross-attention to kv_hidden)."""
  attn = attention.self
  n_heads = attn.num_attention_heads
  kv_hidden = hidden if kv_hidden is None else kv_hidden
  q = _split_heads(attn.query(hidden), n_heads)
  k, v = (_split_heads(f(kv_hidden), n_heads) for f in (attn.key, attn.value))
  context = relative_attention(q, k, v, attn.distance_embedding, **kwargs)
  return attention.output(_merge_heads(context), hidden)


def _embeddings(bert, inputs_embeds):
  # explicit token_type_ids: BertEmbeddings' default buffer only covers
  # max_position_embeddings (1024) positions
  token_type_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
  return bert.embeddings(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids)


def encode(model, z, desc_bar_ids=None, max_distance=255, impl='compiled'):
  """Seq2SeqModule.encode, for description_flavor 'description'."""
  assert model.description_flavor == 'description'
  z_emb = model.desc_in(z)
  if desc_bar_ids is not None:
    z_emb = z_emb + model.bar_embedding(desc_bar_ids)
  bert = model.transformer.encoder
  hidden = _embeddings(bert, z_emb)
  for layer in bert.encoder.layer:
    hidden = _attention_block(layer.attention, hidden, max_distance=max_distance, impl=impl)
    hidden = layer.output(layer.intermediate(hidden), hidden)
  return hidden


def decode(model, x, bar_ids=None, position_ids=None, encoder_hidden_states=None,
           max_distance=255, window=None, impl='compiled'):
  """Seq2SeqModule.decode: logits for every position."""
  x_emb = model.in_layer(x)
  if bar_ids is not None:
    x_emb = x_emb + model.bar_embedding(bar_ids)
  if position_ids is not None:
    x_emb = x_emb + model.pos_embedding(position_ids)
  L = x.shape[1]
  R = encoder_hidden_states.shape[1]
  if R < L:
    # Seq2SeqModule.decode zero-pads the encoder states to the decoder length, and the
    # padding is attended to
    encoder_hidden_states = F.pad(encoder_hidden_states, (0, 0, 0, L - R))
  bert = model.transformer.decoder.bert
  hidden = _embeddings(bert, x_emb)
  for layer in bert.encoder.layer:
    hidden = _attention_block(
      layer.attention, hidden, causal=True, max_distance=max_distance, window=window,
      impl=impl)
    hidden = _attention_block(
      layer.crossattention, hidden, encoder_hidden_states, max_distance=max_distance,
      impl=impl)
    hidden = layer.output(layer.intermediate(hidden), hidden)
  return model.out_layer(hidden)


def forward(model, x, z, bar_ids=None, position_ids=None, desc_bar_ids=None,
            max_distance=255, window=None, impl='compiled'):
  """Seq2SeqModule.forward."""
  encoder_hidden = encode(model, z, desc_bar_ids, max_distance=max_distance, impl=impl)
  return decode(model, x, bar_ids, position_ids, encoder_hidden,
                max_distance=max_distance, window=window, impl=impl)


def get_loss(model, batch, **kwargs):
  """Seq2SeqModule.get_loss, for description_flavor 'description'. kwargs: as for
  forward."""
  logits = forward(
    model, batch['input_ids'], batch['description'], batch['bar_ids'],
    batch['position_ids'], batch['desc_bar_ids'], **kwargs)
  return model.loss_fn(logits.view(-1, logits.shape[-1]), batch['labels'].reshape(-1))
