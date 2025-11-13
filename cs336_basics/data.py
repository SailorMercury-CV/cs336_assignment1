import numpy as np
import torch
import numpy.typing as npt


def get_batch(
  dataset: npt.NDArray,
  batch_size: int,
  context_length: int,
  device: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """
  dataset: 1D np.array of token ids.
  returns x, y: (B, T) long tensors on device.
  """
  assert dataset.ndim == 1
  # We need room for context_length inputs + 1 label.
  max_start = dataset.shape[0] - (context_length + 1)
  if max_start <= 0:
    raise ValueError("Dataset too small for requested context_length")
  
  starts = np.random.randint(0, max_start + 1, size=(batch_size,))
  x = np.stack([dataset[s : s + context_length] for s in starts], axis=0)
  y = np.stack([dataset[s + 1: s+ 1 + context_length] for s in starts], axis=0)

  x = torch.as_tensor(x, dtype=torch.long, device=device)
  y = torch.as_tensor(y, dtype=torch.long, device=device)

  return x, y

