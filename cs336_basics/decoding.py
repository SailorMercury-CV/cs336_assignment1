import torch


def softmax_with_temperature(
  logits: torch.Tensor,
  temperature: float = 1.0,
) -> torch.Tensor:
  """
  logits: (..., vocab_size)
  
  Returns probs with same shape, last dim normalized.
  """
  if temperature <= 0.0:
    raise ValueError(f"temperature must be > 0, got {temperature}")
  
  scaled = logits / temperature
  # Manual softmax to avoid torch.nn.functional.
  max_vals, _ = scaled.max(dim=-1, keepdim=True)
  exp_scaled = (scaled - max_vals).exp()
  probs = exp_scaled / exp_scaled.sum(dim=-1, keepdim=True)
  return probs


def sample_top_p(
  probs: torch.Tensor,
  top_p: float = 1.0,
) -> torch.Tensor:
  """
  probs: (batch_size, vocab_size).
  returns: (batch_size, 1) sampled token indices using nucleus (top-p) sampling.
  If top_p >= 1.0, this reduces to sampling from the full distribution.
  """
  if not (0.0 < top_p <= 1.0):
    raise ValueError(f"top_p must be in (0, 1], got {top_p})")
  
  if top_p >= 1.0:
    # plain categorical sampling.
    return torch.multinomial(probs, num_samples=1)
  
  # sort probs descending.
  sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
  cumulative = torch.cumsum(sorted_probs, dim=-1)

  # mask tokens beyond mucleus.
  nucleus_mask = cumulative > top_p
  # always keep at least the highest-prob tokens
  nucleus_mask[:, 0] = False

  # zero out probs outside nucleus and renormalize.
  sorted_probs = sorted_probs.masked_fill(nucleus_mask, 0.0)
  sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

  # sample in sorted space, then map back to original indices.
  sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
  sampled_token = sorted_indices.gather(-1, sampled_sorted_idx)
  return sampled_token


def generate_tokens(
  model: torch.nn.Module,
  prefix_ids: torch.Tensor,
  max_new_tokens: int,
  eos_token_id: int,
  temperature: float = 1.0,
  top_p: float = 1.0,
  device: str = "cuda:0",
) -> torch.Tensor:
  """
  Autoregressively generate token IDs from a trained LM.

  - model: Transformer LM, output shape (B, T, V).
  - prefix_ids: initial prompt token IDs (B, T_start).
  - max_new_tokens: maximum number of new tokens to generate.
  - eos_token_id: stop when this token is generated (per sequence).
  - temperature: softmax temperature.
  - top_p: nucleus sampling threshold.
  - device: device string, e.g. "cuda:0" or "cpu".

  returns: Tensor of shape (B, T_start + <=max_new_tokens)
  """
  model_was_training = model.training
  model.eval()

  idx = prefix_ids.to(device)
  batch_size = idx.shape[0]

  # track which sequences have already finished.
  finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

  for _ in range(max_new_tokens):
    # forward pass: (B, T, W)
    logits = model(idx)
    # we only need the last time-step logits.
    next_logits = logits[:, -1, :] # (B, V)

    probs = softmax_with_temperature(next_logits, temperature=temperature)
    next_token = sample_top_p(probs, top_p=top_p)

    # for already-finished sequences, force them to stay at eos.
    next_token = torch.where(
      finished.unsqueeze(-1),
      torch.full_like(next_token, eos_token_id),
      next_token,
    )

    # append to sequence.
    idx = torch.cat([idx, next_token], dim=1)

    # update finished mask.
    finished = finished | (next_token.squeeze(-1) == eos_token_id)

    # if all finished, we can stop early.
    if finished.all():
      break
    
    return idx


def generate_text(
  model: torch.nn.Module,
  tokenizer,
  prompt: str,
  max_new_tokens: int = 128,
  temperature: float = 1.0,
  top_p: float = 1.0,
  device: str = "cuda:0",
  add_bos: bool = False,
) -> str:
  """
  High-level helper: takes a string prompt, returns a string completion.
  """
  eos_token_id = tokenizer.vocab_inv["<|endoftext|>"] # or however you store it

  # encode prompt -> ids
  prompt_ids = tokenizer.encode(prompt)
  if add_bos and "<|bos|>" in tokenizer.vocab_inv:
    bos_id = tokenizer.vocab_inv["<|bos|>"]
    
