from pathlib import Path
import numpy as np
import numpy.typing as npt


_DTYPE_MAP = {
  "uint16": np.uint16,
  "int32": np.int32,
  "int64": np.int64,
}

def _open_memmap(path: str | Path, dtype: np.dtype, mode: str = "r") -> npt.NDArray:
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(p)
  itemsize = np.dtype(dtype).itemsize
  size_bytes = p.stat().st_size
  if size_bytes % itemsize != 0:
    raise ValueError(
      f"File size {size_bytes} not divisible by dtype size {itemsize} "
      f"(path={p}, dtype={dtype})"
    )
  length = size_bytes // itemsize
  return np.memmap(p, dtype=dtype, mode=mode, shape(length,))


def data_loading(
  train_path: str | Path,
  val_path: str | Path,
  dtype: str="unit16",
  ) -> tuple[npt.NDArray, npt.NDArray]:
  """
  Open train/val token streams as memory-mapped 1D arrays.
  dtype must match how you saved the .bin (commonly 'uint16' for vocab<=65535).
  Returns (train_tokens, val_tokens).
  """
  if dtype not in _DTYPE_MAP:
    raise ValueError(f"Unsupported dtype '{dtype}'. Choose from {list(_DTYPE_MAP)}")
  np_dtype = _DTYPE_MAP[dtype]
  train_tokens = _open_memmap(train_path, np_dtype, mode="r")
  val_tokens = _open_memmap(val_path, np_dtype, mode="r")
  return train_tokens, val_tokens

