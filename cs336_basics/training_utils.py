import torch
from typing import Iterable


def clip_gradients(
  parameters: Iterable[torch.nn.Parameter],
  max_l2_norm: float,
) -> None:
  if max_l2_norm <= 0:
    raise ValueError("max_l2_norm must be positive")
  
  params = [p for p in parameters if p.grad is not None]
  if not params:
    return
  
  # compute total L2 norm of all grads
  total_sq = torch.zeros((), device=params[0].grad.device, dtype=torch.float32)
  for p in params:
    grad = p.grad.detach()
    total_sq += grad.float().pow(2).sum()
  
  total_norm = total_sq.sqrt()

  if total_norm > max_l2_norm:
    scale = max_l2_norm / (total_norm + 1e-6)
    for p in params:
      p.grad.mul_(scale)
  
  return
