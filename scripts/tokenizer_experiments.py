import argparse, json, re, time, numpy as np
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer


EOT = "<|endoftext|>"

def load_vocab_merges(vocab_json, merges_json):
  with open(vocab_json) as f:
    v = {int(k): bytes.fromhex(h) for k, h in json.load(f).items()}
  with open(merges_json) as f:
    m = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in json.load(f)]
  return v, m


def sample_docs(path, k=10):
  # Split on EOT; skip empties; return first k non-empty docs.
  txt = Path(path).read_text(encoding="utf-8")
  docs = [d for d in txt.split(EOT) if d.strip()]
  return docs[:k]

def bytes_per_token(tokenizer, docs):
  total_bytes = sum(len(d.encode("utf-8")) for d in docs)
  total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
  return total_bytes / max(total_tokens, 1)

def time_throughput(tokenizer, path, seconds=10):
  # Stream with encode_iterable for fair throughput estimate.
  start = time.time()
  nbytes = 0
  ntoks = 0
  with open(path, "r", encoding="utf-8") as f:
    for tid in tokenizer.encode_iterable(f):
      ntoks += 1
  # More accurate: just read total file size on disk.
  nbytes = Path(path).stat().st_size
  dur = max(time.time() - start, 1e-9)
  return nbytes/dur, ntoks/dur, dur

def save_uint16(ids, out_path):
  arr = np.array(ids, dtype=np.uint16)
  np.save(out_path, arr)

  return int(arr.max())

def tokenize_corpus(tokenizer, in_path, out_path):
  ids = []
  with open(in_path, "r", encoding="utf-8") as f:
    for tid in tokenizer.encode_iterable(f):
      ids.append(tid)
  max_id = save_uint16(ids, out_path)
  return len(ids), max_id

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tinystories-vocab", required=True)
  ap.add_argument("--tinystories-merges", required=True)
  ap.add_argument("--ts_train", default="data/TinyStoriesV2-GPT4-train.txt")
  ap.add_argument("--ts_valid", default="data/TinyStoriesV2-GPT4-valid.txt")
  args = ap.parse_args()

  ts_vocab, ts_merges = load_vocab_merges(args.tinystories_vocab, args.tinystories_merges)

  ts_tok = Tokenizer(ts_vocab, ts_merges, [EOT])
  ts_docs = sample_docs(args.ts_train)
  ts_on_ts = bytes_per_token(ts_tok, ts_docs)

  Path("bin").mkdir(exist_ok=True, parents=True)
  n_ts_train, max_ts_train = tokenize_corpus(ts_tok, args.ts_train, "bin/tinystories_train_uint16.npy")
  n_ts_valid, max_ts_valid = tokenize_corpus(ts_tok, args.ts_valid, "bin/tinystories_valid_uint16.npy")

  print(f"[saved] TinyStories train/valid tokens: {n_ts_train}/{n_ts_valid} (max ids {max_ts_train}/{max_ts_valid})")
  print("Why uint16? Because vocab_size ≤ 65,535 so all token IDs fit, halving disk vs uint32 as the handout suggests.")


if __name__ == "__main__":
  main()