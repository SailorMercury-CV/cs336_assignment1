from dataclasses import dataclass
from typing import Callable, Optional
import time
import torch
from torch import nn

from cs336_basics.data import get_batch
from cs336_basics.losses import cross_entropy
from cs336_basics.adamw import AdamW
from cs336_basics.schedules import get_lr_cosine_schedule
from cs336_basics.train_utils import clip_gradients


@dataclass
class TrainConfig:
  device: str = "cuda:0"
  batch_size: int = 64
  context_length: int = 256
  max_iters: int = 20_000
  eval_interval: int = 500
  eval_batches: int = 50
  max_lr: float = 3e-4
  min_lr: float = 3e-5
  warmup_iters: int = 1_000
  cosine_cycle_iters: int = 19_000
  weight_decay: float = 0.1
  max_grad_norm: float = 1.0
  seed: int = 1337
  log_fn: Optional[Callable[[int, float, Optional[float], float], None]] = None


def _set_seed(seed: int):
  import random
  import numpy as np
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loss(
  model: nn.Module,
  data: "np.ndarray",
  batch_size: int,
  context_length: int,
  device: str,
  num_batches: int = 50,
) -> float:
  model.eval()
  losses = []
  for _ in range(num_batches):
    x, y = get_batch(data, batch_size, context_length, device)
    logits = model(x)
    B, T, V = logits.shape
    loss = cross_entropy(logits.view(B * T, V), y.view(B * T))
    losses.append(loss.item())
  model.train()
  return sum(losses) / max(1, len(losses))


def train_transformer_lm(
  model: nn.Module,
  train_data: "np.ndarray",
  val_data: "np.ndarray",
  cfg: TrainConfig,
) -> None:
  _set_seed(cfg.seed)
  device = cfg.device
  model.to(device)
  opt = AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)

  iter_num, t0 = 0, time.time()

  while iter_num < cfg.max_iters:
    lr = get_lr_cosine_schedule(
      it=iter_num,
      max_learning_rate=cfg.max_lr,
      min_learning_rate=cfg.min_lr,
      warmup_iters=cfg.warmup_iters,
      cosine_cycle_iters=cfg.cosine_cycle_iters,
    )

    for pg in opt.param_groups: pg["lr"] = lr

    x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, device)
    logits = model(x)
    B, T, V = logits.shape
    loss = cross_entropy(logits.view(B * T, V), y.view(B * T))

    opt.zero_grad(set_to_none=True)
    loss.backward()
    clip_gradients(model.parameters(), cfg.max_grad_norm)
    opt.step()
    iter_num += 1

    if (iter_num % cfg.eval_interval == 0) or (iter_num == 1):
      val_loss = evaluate_loss(model, val_data, cfg.batch_size, cfg.context_length, device, cfg.eval_batches)
      if cfg.log_fn:
        cfg.log_fn(iter_num, loss.item(), val_loss, lr)
      else:
        dt = time.time() - t0
        print(f"[it {iter_num:6d}] train={loss.item():.4f} val={val_loss:.4f} lr={lr:.2e} dt={dt:.1f}s")
      t0 = time.time()