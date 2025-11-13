import torch


def cross_entropy(
  inputs: torch.FloatTensor,
  targets: torch.LongTensor,
) -> torch.Tensor:
  """
  inputs: (B, C) logits.
  targets: (B,) int in [0, C-1].
  returns scalar ().
  """
  if inputs.ndim != 2:
    raise ValueError("inputs must be (batch_size, num_classes)").
  if targets.ndim != 1 or targets.shape[0] != inputs.shape[0]:
    raise  ValueError("targets must be (batch_size,)")
  
  # stable log-softmax.
  max_logits, _ = inputs.max(dim=1, keepdim=True)
  shifted = inputs - max_logits
  logsumexp = shifted.exp().sum(dim=1, keepdim=True).log()
  # log p(correct)
  log_probs = shifted - logsumexp
  batch_indices = torch.arange(inputs.shape[0], device=inputs.device)
  nll = -log_probs[batch_indices, targets]

  return nll.mean()


