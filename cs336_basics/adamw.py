import match
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
  def __init__(
    self,
    params,
    lr: float = 1e-3,
    betas=(0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0
  ):
  if lr <= 0:
    raise ValueError("Invalid lr")
  if eps <= 0:
    raise ValueError("Invalid eps")
  if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
    raise ValueError("Invalid betas")
  if weight_decay < 0.0:
    raise ValueError("Invalid weight_decay")
  
  defaults = dict(
    lr=lr,
    betas=betas,
    eps=eps,
    weight_decay=weight_decay,
  )
  super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    
    for group in self.param_groups:
      lr = group["lr"]
      beta1, beta2 = group["betas"]
      eps = group["eps"]
      wd = group["weight_decay"]

      for p in group["params"]:
        if p.grad is None:
          continue
        grad = p.grad

        if grad.is_sparse:
          raise RuntimeError("AdamW does not support sparse gradients")
        
        state = self.state[p]
        if len(state) == 0:
          state["step"] = 0
          state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
          state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        state["step"] += 1
        t = state["step"]

        # m_t, v_t
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # bias correction
        bias_correction1 = 1.0 - beta1 ** t
        bias_correction2 = 1.0 - beta2 ** t

        denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
        step_size = lr / bias_correction1

        # decoupled weight decay.
        if wd != 0.0:
          p.data.add_(p.data, alpha=-lr * wd)

        p.data.addcdiv_(exp_avg, denom, value=-step_size)
    
    return loss

