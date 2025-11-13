import os
from typing import BinaryIO, IO
import torch
from torch.nn import Module
from torch.optim import Optimizer


CKPT_VERSION = 1

def save_checkpoint(
  model: Module,
  optimizer: Optimizer,
  iteration: int,
  out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
  payload = {
    "version": CKPT_VERSION,
    "iteration": int(iteration),
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
  }
  torch.save(payload, out)


def load_checkpoint(
  src: str | os.PathLike | BinaryIO | IO[bytes],
  model: Module,
  optimizer: Optimizer,
) -> int:
  payload = torch.load(src, map_location="cpu")
  if not isinstance(payload, dict) or "model" not in payload or "optimizer" not in payload:
    raise ValueError("Invalid checkpoint format")
  model.load_state_dict(payload["model"])
  optimizer.load_state_dict(payload["optimizer"])
  return int(payload.get("iteration", 0))