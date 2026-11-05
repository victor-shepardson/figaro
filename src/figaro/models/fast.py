"""Faster inference for Seq2SeqModule, using the weights of its HF Bert encoder/decoder.

- on CPU, relative_attention computes the relative_key_query attention scores with two
  matmuls against the distance embeddings plus a gather, instead of HF's einsum against
  a (L, R, d) positional embedding tensor (on MPS, the einsum is faster)
- the decoder input isn't padded to the encoder length (the causal mask keeps padding
  from affecting real positions anyway)
- the last layer only computes the last position, when only next-token logits are needed
- FastDecoder optionally caches self-attention keys/values with a strided window: the
  window grows to context_size tokens, then restarts with the last min_context tokens

Without the cache, results match Seq2SeqModule.encode/decode up to float error.
With it, the model sees between min_context and context_size tokens of context instead
of always context_size, and the description scrolls only when the window restarts.
"""
import math

import torch
import torch.nn.functional as F


def _split_heads(x, n_heads):
  b, n, d = x.shape
  return x.view(b, n, n_heads, d // n_heads).transpose(1, 2)


def _merge_heads(x):
  b, h, n, d = x.shape
  return x.transpose(1, 2).reshape(b, n, h * d)


def relative_attention(q, k, v, distance_embedding, q_start=0, causal=False, key_scores=None):
  """Attention with relative_key_query position scores, as in HF BertSelfAttention.

  q: (b, h, L, d) at positions q_start..q_start+L-1; k, v: (b, h, R, d) at positions 0..R-1.
  key_scores: optional precomputed key_relative_scores for query positions 0..
  """
  b, h, L, d = q.shape
  R = k.shape[2]
  table, index = _distance_table(distance_embedding, q_start, L, R)
  scores = q @ k.transpose(-1, -2)
  if q.device.type == 'cpu':
    scores = scores + (q @ table.T).gather(-1, index.expand(b, h, L, R))
  else:
    # as in HF, which is faster on MPS
    scores = scores + torch.einsum('bhld,lrd->bhlr', q, table[index])
  if key_scores is None:
    scores = scores + _key_scores(k, table, index)
  else:
    scores = scores + key_scores[:, :, q_start:q_start + L]
  scores = scores / math.sqrt(d)
  if causal:
    scores = scores.masked_fill(index < R - 1 - q_start, float('-inf'))  # k_pos > q_pos
  return F.softmax(scores, dim=-1) @ v


def _distance_table(distance_embedding, q_start, L, R):
  """The distance embeddings for queries at q_start..q_start+L-1 and keys at 0..R-1,
  and (L, R) indices of each query/key pair's distance in it."""
  max_pos = (distance_embedding.num_embeddings + 1) // 2
  # distances q_pos - k_pos range from d_min to q_start + L - 1
  d_min = q_start - (R - 1)
  table = distance_embedding.weight[d_min + max_pos - 1 : q_start + L + max_pos - 1]
  q_pos = torch.arange(q_start, q_start + L, device=table.device)
  k_pos = torch.arange(R, device=table.device)
  return table, q_pos[:, None] - k_pos[None, :] - d_min


def _key_scores(k, table, index):
  """Key-side relative position scores (b, h, L, R)."""
  b, h, R, d = k.shape
  L = index.shape[0]
  if k.device.type == 'cpu' and L > 4:
    return (k @ table.T).gather(-1, index.T.expand(b, h, R, L)).transpose(-1, -2)
  elif k.device.type == 'cpu':
    # few queries: only the needed distances
    return (k.unsqueeze(2) * table[index]).sum(-1)
  else:
    return torch.einsum('bhrd,lrd->bhlr', k, table[index])


def key_relative_scores(k, distance_embedding, n_queries):
  """Key-side relative position scores of keys k (b, h, R, d) for query positions
  0..n_queries-1, which can be passed to relative_attention when the keys are fixed."""
  table, index = _distance_table(distance_embedding, 0, n_queries, k.shape[2])
  return _key_scores(k, table, index)


def encode(model, z, desc_bar_ids=None):
  """Same as Seq2SeqModule.encode, for description_flavor 'description'."""
  assert model.description_flavor == 'description'
  z_emb = model.desc_in(z)
  if desc_bar_ids is not None:
    z_emb = z_emb + model.bar_embedding(desc_bar_ids)
  bert = model.transformer.encoder
  hidden = bert.embeddings(inputs_embeds=z_emb)
  for layer in bert.encoder.layer:
    attn = layer.attention.self
    n_heads = attn.num_attention_heads
    q, k, v = (_split_heads(f(hidden), n_heads) for f in (attn.query, attn.key, attn.value))
    context = relative_attention(q, k, v, attn.distance_embedding)
    hidden = layer.attention.output(_merge_heads(context), hidden)
    hidden = layer.output(layer.intermediate(hidden), hidden)
  return hidden


def cross_keys_values(model, encoder_hidden_states, n_queries=None):
  """Cross-attention (key, value, key_scores) heads of each decoder layer, for encoder 
  states, with key-side relative scores for decoder positions 0..n_queries-1 (default:
  context_size)."""
  n_queries = n_queries or model.context_size
  kv = []
  for layer in model.transformer.decoder.bert.encoder.layer:
    attn = layer.crossattention.self
    n_heads = attn.num_attention_heads
    k, v = (_split_heads(f(encoder_hidden_states), n_heads) for f in (attn.key, attn.value))
    kv.append((k, v, key_relative_scores(k, attn.distance_embedding, n_queries)))
  return kv


def embed(model, x, bar_ids=None, position_ids=None):
  """Decoder input embeddings, as in Seq2SeqModule.decode."""
  x_emb = model.in_layer(x)
  if bar_ids is not None:
    x_emb = x_emb + model.bar_embedding(bar_ids)
  if position_ids is not None:
    x_emb = x_emb + model.pos_embedding(position_ids)
  return x_emb


def decoder_layers(model, x_emb, cross_kv, past_kv=None, last_only=False):
  """Run the decoder layers on new tokens following past_kv (None: no past tokens).

  x_emb: (b, n, d) input embeddings of the new tokens, at window positions
    past_len..past_len+n-1
  cross_kv: from cross_keys_values
  past_kv: per layer (key, value) heads of the past tokens, as returned by this function
  last_only: compute the output only for the last new token
  Returns (hidden, kv): last layer outputs (b, n or 1, d), and per layer (key, value)
  heads of past and new tokens.
  """
  bert = model.transformer.decoder.bert
  layers = bert.encoder.layer
  past_len = 0 if past_kv is None else past_kv[0][0].shape[2]
  hidden = bert.embeddings(inputs_embeds=x_emb)
  kv = []
  for i, layer in enumerate(layers):
    attn = layer.attention.self
    n_heads = attn.num_attention_heads
    q, k, v = (_split_heads(f(hidden), n_heads) for f in (attn.query, attn.key, attn.value))
    if past_kv is not None:
      k = torch.cat([past_kv[i][0], k], dim=2)
      v = torch.cat([past_kv[i][1], v], dim=2)
    kv.append((k, v))
    q_start = past_len
    if last_only and i == len(layers) - 1:
      # keys/values of all new tokens are still needed (and cached)
      q, hidden = q[:, :, -1:], hidden[:, -1:]
      q_start = past_len + x_emb.shape[1] - 1
    context = relative_attention(q, k, v, attn.distance_embedding, q_start, causal=True)
    hidden = layer.attention.output(_merge_heads(context), hidden)

    attn = layer.crossattention.self
    q = _split_heads(attn.query(hidden), attn.num_attention_heads)
    k, v, key_scores = cross_kv[i]
    context = relative_attention(q, k, v, attn.distance_embedding, q_start, key_scores=key_scores)
    hidden = layer.crossattention.output(_merge_heads(context), hidden)

    hidden = layer.output(layer.intermediate(hidden), hidden)
  return hidden, kv


def decode(model, x, bar_ids=None, position_ids=None, encoder_hidden_states=None):
  """Same as Seq2SeqModule.decode (logits for every position), with cross-attention."""
  hidden, _ = decoder_layers(
    model, embed(model, x, bar_ids, position_ids),
    cross_keys_values(model, encoder_hidden_states, x.shape[1]))
  return model.out_layer(hidden)


def description_encoder(model, description, desc_bar_ids):
  """Returns encode(first_bars) -> encoder hidden states, with the description window
  starting at each sequence's decoder window's first bar, zero-padded to the context
  size, as in Seq2SeqModule.sample. Results are memoized.

  description, desc_bar_ids: (b, n) description token ids and bar ids
  """
  n_ctx = model.context_size
  memo = {}
  def encode_window(first_bars):
    key = tuple(first_bars.tolist())
    if key not in memo:
      batch_size = description.shape[0]
      z = torch.zeros(batch_size, n_ctx, dtype=torch.long)
      z_bar_ids = torch.zeros(batch_size, n_ctx, dtype=torch.long)
      for j in range(batch_size):
        indices = torch.nonzero(desc_bar_ids[j].cpu() == first_bars[j].cpu())
        idx = indices[0, 0].item() if indices.size(0) > 0 else description.shape[1] - 1
        length = min(n_ctx, description.shape[1] - idx)
        z[j, :length] = description[j, idx:idx + length]
        z_bar_ids[j, :length] = desc_bar_ids[j, idx:idx + length]
      memo.clear()  # only the latest window is needed
      memo[key] = encode(model, z.to(model.device), z_bar_ids.to(model.device))
    return memo[key]
  return encode_window


class FastDecoder:
  """Next-token logits for a token sequence that grows (and may be truncated) between
  calls.

  encode: function from the first bar id of each sequence's decoder window (b,) to
    encoder hidden states (b, R, d), e.g. from description_encoder
  cache: if False, every call recomputes the last context_size tokens (a sliding
    window, as in Seq2SeqModule.sample). If True, keys/values are cached in a strided
    window: once it exceeds context_size tokens, it restarts with the last min_context
  min_context: defaults to context_size // 2
  """
  def __init__(self, model, encode, cache=True, min_context=None):
    self.model = model
    self.encode = encode
    self.cache = cache
    self.n_ctx = model.context_size
    self.min_context = min_context or self.n_ctx // 2
    assert 0 < self.min_context < self.n_ctx
    self.n_forward = 0  # forward passes, for diagnostics
    self.n_window_starts = 0
    self._clear(0)

  def _clear(self, start):
    self.start = start  # index of the first token of the cached window
    self.history = None  # (3, b, T) ids, bar ids, position ids the cache was built from
    self.kv = None
    self.first_bars = None
    self.cross_kv = None

  def _cross_kv(self, first_bars):
    if self.first_bars is None or not torch.equal(first_bars, self.first_bars):
      self.first_bars = first_bars
      self.cross_kv = cross_keys_values(self.model, self.encode(first_bars))
    return self.cross_kv

  @torch.no_grad()
  def logits(self, ids, bar_ids, position_ids):
    """Logits (b, vocab) for the token following ids, bar_ids, position_ids (b, T)."""
    model = self.model
    device = model.device
    history = torch.stack([
      torch.as_tensor(t, dtype=torch.long).to(device) for t in (ids, bar_ids, position_ids)])
    T = history.shape[-1]

    if not self.cache:
      window = history[..., -self.n_ctx:]
      self.n_forward += 1
      hidden, _ = decoder_layers(
        model, embed(model, *window), self._cross_kv(window[1, :, 0]), last_only=True)
      return model.out_layer(hidden[:, -1])

    # length of the prefix shared with the cached history
    if self.history is None:
      shared = 0
    else:
      n = min(T, self.history.shape[-1])
      mismatch = (history[..., :n] != self.history[..., :n]).flatten(0, 1).any(0)
      shared = mismatch.nonzero()[0, 0].item() if mismatch.any() else n
    if shared <= self.start or T - self.start > self.n_ctx:
      self._clear(0 if T <= self.n_ctx else T - self.min_context)
      self.n_window_starts += 1
      shared = self.start
    # recompute at least the last token, whose logits weren't kept
    n_past = min(shared, T - 1) - self.start
    past_kv = None
    if n_past > 0:
      past_kv = [(k[:, :, :n_past], v[:, :, :n_past]) for k, v in self.kv]
    window = history[..., self.start + n_past:]
    self.n_forward += 1
    hidden, self.kv = decoder_layers(
      model, embed(model, *window), self._cross_kv(history[1, :, self.start]),
      past_kv, last_only=True)
    self.history = history
    return model.out_layer(hidden[:, -1])
