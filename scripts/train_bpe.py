import json, time, argparse, os, psutil
from pathlib import Path

from cs336_basics.tokenizer import train_bpe

def bytestr(b):
  # pretty-print the longest token as a safe string.
  try:
    return b.decode('utf-8', errors='replace')
  except Exception:
    return str(list(b))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--input", required=True)
  ap.add_argument("--vocab-size", type=int, required=True)
  ap.add_argument("--special", nargs="*", default=["<|endoftext|>"])
  ap.add_argument("--outdir", default="tokenizers")
  args = ap.parse_args()

  Path(args.outdir).mkdir(parents=True, exist_ok=True)

  t0 = time.time()
  process = psutil.Process(os.getpid())
  vocab, merges = train_bpe(args.input, args.vocab_size, args.special)
  wall = time.time() - t0
  mem_gb = process.memory_info().rss / (1024**3)

  # Serialize (bytes -> hex to keep JSON clean).
  vocab_json = {int(i): v.hex() for i, v in vocab.items()}
  merges_json = [[a.hex(), b.hex()] for (a, b) in merges]

  base = f"bpe_v{args.vocab_size}"
  with open(f"{args.outdir}/{base}.vocab.json", "w") as f:
    json.dump(vocab_json, f)
  with open(f"{args.outdir}/{base}.merges.json", "w") as f:
    json.dump(merges_json, f)

  # Longest token (by length in bytes).
  longest = max(vocab.values(), key=len)
  print(f"[done] wall={wall:.1f}s rss~{mem_gb:.1f}GB longest_token_bytes={len(longest)} example={bytestr(longest)!r}")


if __name__ == "__main__":
  main()