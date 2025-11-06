import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
  """
  Implements Root Mean Square Normalization (Problem 3.4.1).
  """
  def __init__(self, d_model: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    # g_i, the learnable gain parameter, initialized to 1.
    self.weight = nn.Parameter(torch.ones(d_model))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    return (x / rms) * self.weight


def gelu(x: torch.Tensor) -> torch.Tensor:
  """
  Implements the GELU activation function (Problem 3.4.2a).
  """
  return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionWiseFeedForward(nn.Module):
  """
  Implements the Position-Wise Feed-Forward Network (Problem 3.4.2b).
  """
  def __init__(self, d_model: int, d_ff: int):
    super().__init__()

    self.w1 = nn.Linear(d_model, d_ff, bias=False)
    self.w2 = nn.Linear(d_ff, d_model, bias=False)

  def forward(self, x):
    # FFN(x) = GELU(X * W_1) * W2.
    return self.w2(gelu(self.w1(x)))


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """
  Implements the numerically stable softmax function (Problem 3.4.3).
  """
  max_val = torch.max(x, dim=dim, keepdim=True).values
  exps = torch.exp(x - max_val)

  return exps / torch.sum(exps, dim=dim, keepdim=True)


def scaled_dot_product_attention(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  mask: torch.Tensor = None,
  dropout: nn.Module = None
) -> torch.Tensor:
  """
  Implements Scaled Dot-Product Attention (Problem 3.4.3).
  """
  # Q, K, V are expected to be (B, num_heads, T, d_k/d_v).
  d_k = q.size(-1)

  # QK^T / sqrt(d_k).
  scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

  # Apply mask if provided.
  # Add -inf to masked positions before softmax.
  if mask is not None:
    scores = scores.masked_fill(mask, -torch.inf)
  
  # attn_probs = softmax(scores).
  attn_probs = softmax(scores, dim=-1)

  # Apply attention dropout.
  if dropout is not None:
    attn_probs = dropout(attn_probs)

  return attn_probs @ v


class CausalMultiHeadSelfAttention(nn.Module):
  """
  Implements Causal Multi-Head Self-Attention (Problem 3.4.4).
  """
  def __init__(
    self,
    d_model: int,
    num_heads: int,
    attn_pdrop: float
  ):
    super().__init__()
    assert d_model % num_heads == 0

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    # We use a single Linear layer for all head projections,
    # which is more efficient.
    self.q_proj = nn.Linear(d_model, d_model, bias=False)
    self.k_proj = nn.Linear(d_model, d_model, bias=False)
    self.v_proj = nn.Linear(d_model, d_model, bias=False)

    # W^O, the output projection.
    self.o_proj = nn.Linear(d_model, d_model, bias=False)

    self.attn_dropout = nn.Dropout(attn_pdrop)


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # B = batch_size, T = sequence_length, C = d_model
    B, T, C = x.shape

    # 1. Project Q, K, V.
    # (B, T, C) -> (B, T, num_heads, d_k) -> (B, num_heads, T, d_k).
    q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    # 2. Create causal mask.
    # We use torch.triu to create an upper-triangular matrix.
    # The mask should be (T, T).
    causal_mask = torch.triu(
      torch.ones(T, T, device=x.device, dtype=torch.bool),
      diagonal= 1
    )
    # causal_mask shape is (T, T), which will be broadcasted to (B, nh, T, T)
    # by the scaled_dot_product_attention function.

    # 3. Apply scaled dot-product attention.
    attn_output = scaled_dot_product_attention(
      q, k, v,
      mask=causal_mask,
      dropout=self.attn_dropout
    )

    # 4. Concat heads and apply output projection.
    # (B, num_heads, T, d_k) -> (B, T, num_heads, d_k) -> (B, T, C).
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

    # MultiHead(Q, K, V) = Concat(heads) * W^O.
    return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
  """
  Implements the Pre-Norm Transformer Block (Problem 3.4).
  """
  def __init__(
    self,
    d_model: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float
  ):
    super().__init__()

    # First sub-layer: Causal Multi-Head Self-Attention.
    self.norm1 = RMSNorm(d_model)
    self.attn = CausalMultiHeadSelfAttention(
      d_model=d_model,
      num_heads=num_heads,
      attn_pdrop=attn_pdrop
    )
    self.dropout1 = nn.Dropout(residual_pdrop)

    # Second sub-layer: Position-Wise Feed-Forward.
    self.norm2 = RMSNorm(d_model)
    self.ffn = PositionWiseFeedForward(
      d_model=d_model,
      d_ff=d_ff
    )
    self.dropout2 = nn.Dropout(residual_pdrop)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Pre-norm architecture.
    x = x + self.dropout1(self.attn(self.norm1(x)))

    x = x + self.dropout2(self.ffn(self.norm2(x)))

    return x


class TransformerLM(nn.Module):
  """
  Implements the full Transformer Language Model (Problem 3.5).
  """
  def __init__(
    self,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    attn_pdrop: float,
    residual_pdrop: float,
    d_ff: int = None
  ):
    super().__init__()

    if d_ff is None:
      # Per section 3.4.2, d_ff is typically 4 * d_model.
      d_ff = 4 * d_model

    self.vocab_size = vocab_size
    self.context_length = context_length
    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads

    # Token and Positional Embeddings.
    # Token Embedding.
    self.token_embedding = nn.Embedding(vocab_size, d_model)

    # Absolute Position Embeddings.
    self.pos_embedding = nn.Embedding(context_length, d_model)

    # Add & Dropout.
    self.emb_dropout = nn.Dropout(residual_pdrop)

    # Transformer Blocks.
    self.blocks = nn.ModuleList([
      TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop
      ) for _ in range(num_layers)
    ])

    # Final Norm.
    # This is required for the pre-norm architecture.
    self.final_norm = RMSNorm(d_model)

    # Output Embedding (Linear).
    self.output_embedding = nn.Linear(d_model, vocab_size, bias=False)

    self.token_embedding.weight = self.output_embedding.weight

  def forward(self, idx: torch.Tensor) -> torch.Tensor:
    B, T = idx.shape

    if T > self.context_length:
      raise ValueError(
        f"Input sequence length ({T}) exceeds "
        f"model context length ({self.context_length})"
      )

    # 1. Get Embeddings.
    # tok_emb is (B, T, d_model).
    tok_emb = self.token_embedding(idx)

    # pos_emb is (T, d_moodel)
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    pos_emb = self.pos_embedding(pos)

    # 2. Add embeddings and apply dropout.
    # (B, T, d_model) + (T, d_model) -> (B, T, d_model).
    x = tok_emb + pos_emb
    x = self.emb_dropout(x)

    # 3. Pass through Transformer Blocks.
    for block in self.blocks:
      x = block(x)
    
    # 4. Final Norm
    x = self.final_norm(x)

    # 5. Output Embedding (Linear layer).
    # (B, T, d_model) -> (B, T, vocab_size)
    logits = self.output_embedding(x)

    # Softmax is applied in the loss function or during generation.
    return logits
  

