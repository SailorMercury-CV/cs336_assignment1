import argparse
import numpy as np
from cs336_basics.train_loop import TrainConfig, train_transformer_lm
from cs336_basics.model import TransformerLM


def open_memmap(path: str, dtype=np.uint16)
  mm = np.memmap(path, dtype=dtype, mode="r")
  return mm


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--train", required=True)
  ap.add_argument("--val", required=True)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--batch_size", type=int, default=64)
  ap.add_argument("--context_length", type=int, default=256)
  ap.add_argument("--max_iters", type=int, default=20_000)
  ap.add_argument("--eval_interval", type=int, default=500)
  ap.add_argument("--eval_batches", type=int, default=50)
  ap.add_argument("--max_lr", type=float, default=3e-4)
  ap.add_argument("--min_lr", type=float, default=3e-5)
  ap.add_argument("--warmup_iters", type=int, default=1_000)
  ap.add_argument("--cosine_iters", type=int, default=19_000)
  ap.add_argument("--wd", type=float, default=0.1)
  ap.add_argument("--max_grad_norm", type=float, default=1.0)
  args = ap.parse_args()

  train_tokens = open_memmap(args.train, dtype=np.uint16)
  val_tokens = open_memmap(args.val, dtype=np.uint16)

  model = TransformerLM(
    vocab_size=50_257,
    d_model=768,
    n_heads=12,
    n_layers=12,
    context_length=args.context_length,
    dropout_p=0.0,
    bias=False,
    use_rmsnorm=True,
    parallel_layers=False,
  )

  cfg = TrainConfig(
    device=args.device,
    batch_size=args.batch_size,
    context_length=args.context_length,
    max_iters=args.max_iters,
    eval_interval=args.eval_interval,
    eval_batches=args.eval_batches,
    max_lr=args.max_lr,
    min_lr=args.min_lr,
    warmup_iters=args.warmup_iters,
    cosine_cycle_iters=args.cosine_iters,
    weight_decay=args.wd,
    max_grad_norm=args.max_grad_norm,
  )

  train_transformer_lm(model, train_tokens, val_tokens, cfg)

if __name__ == "__main__":
  main()