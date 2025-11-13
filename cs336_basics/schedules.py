import math


def get_lr_cosine_schedule(
  it: int,
  max_learning_rate: float,
  min_learning_rate: float,
  warmup_iters: int,
  cosine_cycle_iters: int,
) -> float:
  if it < 0:
    raise ValueError("it must be non-negative")
  if warmup_iters < 0 or cosine_cycle_iters < 0:
    raise ValueError("iters must be non-negative")
  if max_learning_rate < 0 or min_learning_rate < 0:
    raise ValueError("learning rates must be non-negative")
  
  # 1) Warmup
  if warmup_iters > 0 and it < warmup_iters:
    return max_learning_rate * (it / float(warmup_iters))
  
  # Shift iteration after warmup.
  it_after_warmup = it - warmup_iters

  # 2) Cosine decay.
  if cosine_cycle_iters > 0 and it_after_warmup < cosine_cycle_iters:
    # progress in [0, 1]
    progress = it_after_warmup / float(cosine_cycle_iters)
    # cosine in [0, 1]
    cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
    # interpolate between max_lr and min_lr.
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_term
  
  # 3). After schedule, stick at min_lr.
  return min_learning_rate