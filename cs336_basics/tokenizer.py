from typing import Iterable, Iterator, Dict, List, Tuple, Optional
import regex as re
import binascii
import json


# GPT-2 style pre-tokenizer (uses the 'regex' package, not Python's 're')
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# UTF-8 bytes for U+FFFD (replacement character), used if an unknown id sneaks in.
_UTF8_REPLACEMENT = b"\xEF\xBF\xBD"


def get_pair_stats(splits: Dict[bytes, List[bytes]],
                   word_freqs: Dict[bytes, int]) -> Dict[Tuple[bytes, bytes], int]:
    """
    Computes the frequency of each adjacent byte pair from the current splits.
    
    Args:
        splits: A dictionary mapping each pre-token (bytes) to its current list 
                of subword tokens (list[bytes]).
        word_freqs: A dictionary mapping each pre-token (bytes) to its frequency
                    in the corpus.

    Returns:
        A dictionary mapping each adjacent pair (tuple[bytes, bytes]) to its
        total frequency.
    """
    pair_stats = {}
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) < 2:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            pair_stats[pair] = pair_stats.get(pair, 0) + freq
    
    return pair_stats


def merge_byte_list(tokens: List[bytes],
                    pair: Tuple[bytes, bytes],
                    new_token: bytes) -> List[bytes]:
    """
    Applies a single merge rule to a list of byte tokens.

    Args:
        tokens: A list of byte tokens (e.g., [b't', b'h', b'e']).
        pair: The pair to merge (e.g., (b't', b'h')).
        new_token: The resulting merged token (e.g., b'th')
    
    Returns:
        A new list of byte tokens with the pair merged.
    """
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]
              ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Implements the BPE training algorithm as described in Problem (train_bpe).
    
    Args:
        input_path: Path to the training text file[cite: 189].
        vocab_size: The maximum final vocabulary size[cite: 190].
        special_tokens: A list of special token strings to add[cite: 191].

    Returns:
        A tuple containing:
        - vocab: A dict mapping token IDs (int) to token bytes[cite: 194].
        - merges: A list of merged pairs (tuple[bytes, bytes]), 
                  in order of creation [cite: 195-197].
    """
    if vocab_size < 256:
        raise ValueError("Vocabulary size must be at least 256.")
    
    # 1. Initialize vocabulary with 256 base bytes [cite: 126-127].
    vocab = {i: bytes([i]) for i in range(256)}

    # 2. Add special tokens to the vocabulary.
    special_token_bytes_set = set()
    for i, token_str in enumerate(special_tokens):
        token_bytes = token_str.encode("utf-8")
        vocab[256 + i] = token_bytes
        special_token_bytes_set.add(token_bytes)
    
    # 3. Pre-tokenize the corpus and get word frequencies [cite: 132].
    pre_tokenizer_regex = re.compile(GPT2_PAT)
    word_freqs = {}

    # Create the regex to split *by* special tokens, preserving them.
    if special_tokens:
        specials_sorted = sorted(special_tokens, key=len, reverse=True)
        special_pat = f"({'|'.join(re.escape(s) for s in specials_sorted)})"
        special_splitter_regex = re.compile(special_pat)


    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

            if not special_tokens:
                for word in pre_tokenizer_regex.findall(text):
                    word_bytes = word.encode('utf-8')
                    word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1
            else:
                chunks = special_splitter_regex.split(text)

                for i, chunk in enumerate(chunks):
                    if not chunk:
                        continue
                    
                    if i % 2 == 1:
                        # This is a special token.
                        word_bytes = chunk.encode("utf-8")
                        word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1
                    else:
                        for word in pre_tokenizer_regex.findall(chunk):
                            word_bytes = word.encode('utf-8')
                            word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1
    
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return {}, []
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return {}, []
    
    # 4. Initialize splits for each word.
    splits = {}
    for word_bytes in word_freqs.keys():
        if word_bytes in special_token_bytes_set:
            splits[word_bytes] = [word_bytes]
        else:
            splits[word_bytes] = [bytes([b]) for b in word_bytes]
    
    merges = []

    # 5. Compute BPE merges.
    num_merges_needed = vocab_size - len(vocab)

    for i in range(num_merges_needed):
        # 5a. Get statistics for all adjacent pairs.
        pair_stats = get_pair_stats(splits, word_freqs)

        if not pair_stats:
            break
        
        # 5b. Find the most frequent pair.
        best_pair = max(pair_stats, key=lambda p: (pair_stats[p], p))

        # 5c. Create the new merged token.
        new_token_bytes = best_pair[0] + best_pair[1]

        # 5d. Add to merges list and vocabulary.
        merges.append(best_pair)
        new_token_id = len(vocab)
        vocab[new_token_id] = new_token_bytes

        # 5e. Update all splits to reflect the new merge.
        for word in word_freqs.keys():
            splits[word] = merge_byte_list(splits[word], best_pair, new_token_bytes)
    
    return vocab, merges



class Tokenizer:
    """
    Byte-level BPE tokenizer with:
      - regex pretokenization (GPT-2 pattern)
      - merge application in creation order
      - special-token preservation
      - UTF-8 encode/decode with errors='replace' on decode
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None
    ):
        self.vocab = vocab
        # Create the inverse vocabulary for encoding.
        self.inv_vocab = {v: k for k, v in vocab.items()}

        # Create an efficient (pair -> rank) map for merges [cite: 227]
        self.merges = {pair: i for i, pair in enumerate(merges)}

        self.pre_tokenizer = re.compile(GPT2_PAT)

        self.special_tokens = {}
        self.inv_special_tokens = {}

        if special_tokens:
            # 1. Add special tokens to vocab if not present [cite: 254]
            for token_str in special_tokens:
                token_bytes = token_str.encode("utf-8")
                if token_bytes not in self.inv_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token_bytes
                    self.inv_vocab[token_bytes] = new_id
                
                # Store for special splitting logic.
                token_id = self.inv_vocab[token_bytes]
                self.special_tokens[token_str] = token_id
                self.inv_special_tokens[token_id] = token_str
            
            # 2. Create a regex to split by special tokens.
            # This ensure they are preserved as single tokens [cite: 160, 243]
            specials_sorted = sorted(special_tokens, key=len, reverse=True)
            special_pat = f"({'|'.join(re.escape(s) for s in specials_sorted)})"
            self.special_splitter = re.compile(special_pat)
        else:
            self.special_splitter = None


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None
    ) -> "Tokenizer":
        # Load vocabulary.
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_str_keys = json.load(f)
            # Re-encode from latin-1 string hack back to bytes.
            vocab = {int(k): v.encode('latin-1') for k, v in vocab_str_keys.items()}
            # vocab = {int(k): bytes(v) for k, v in vocab_str_keys.items()}

        # Load merges.
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    # Re-encode from latin-1 string hack back to bytes.
                    p1_str, p2_str = line.strip().split()
                    merges.append((p1_str.encode('latin-1'), p2_str.encode('latin-1')))
        
        return cls(vocab, merges, special_tokens)


    def _encode_bytes(self, b: bytes) -> List[int]:
        """
        Helper function to encode a *single* pre-token's bytes.
        Applies BPE merges iteratively in the correct order [cite: 227].
        """
        # Start with individual bytes.
        tokens = [bytes([byte]) for byte in b]

        while True:
            # Find the merge with the *lowest rank* (earliest merge) [cite: 236]
            min_rank = float('inf')
            best_pair_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair_idx = i
            
            # If no applicable merges are found, we're done [cite: 238]
            if best_pair_idx == -1:
                break
            
            # Apply the best merge.
            pair_to_merge = (tokens[best_pair_idx], tokens[best_pair_idx+1])
            new_token = pair_to_merge[0] + pair_to_merge[1]
            tokens = (
                tokens[:best_pair_idx]
                + [new_token]
                + tokens[best_pair_idx+2:]
            )
        
        # Convert the final list of byte tokens to their integer IDs.
        return [self.inv_vocab[t] for t in tokens]


    def encode(self, text: str) -> list[int]:
        """
        Encodes an input text string into a sequence of token IDs [cite: 266].
        """
        token_ids = []

        # 1. Handle special tokens [cite: 243]
        if self.special_splitter:
            # Split the text by special tokens.
            chunks = self.special_splitter.split(text)
        else:
            chunks = [text]
        
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            
            if self.special_splitter and i % 2 == 1:
                # This is a special token.
                token_ids.append(self.special_tokens[chunk])
            else:
                # This is a regular text chunk.
                # 2. Pre-tokenize the chunk [cite: 225].
                for pre_token in self.pre_tokenizer.findall(chunk):
                    # 3. Encode pre-token to bytes (UTF-8) [CITE: 225].
                    pre_token_bytes = pre_token.encode('utf-8')
                    # 4. Apply BPE merges and get IDs [cite: 227]
                    token_ids.extend(self._encode_bytes(pre_token_bytes))
        
        return token_ids

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encodes an iterable of strings, yielding token IDs [cite: 267].
        """
        for text_chunk in iterable:
            yield from self.encode(text_chunk)
    

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a sequence of token IDs back into text [cite: 269].
        """
        # 1. Look up bytes for each ID.
        token_bytes_list = [self.vocab[i] for i in ids]

        # 2. Concatenate all bytes.
        all_bytes = b"".join(token_bytes_list)

        # 3. Decode to UTF-8 string, replacing errors [cite: 251]
        return all_bytes.decode("utf-8", errors="replace")
