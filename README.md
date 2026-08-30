# xor_keyrec_ml

Classifier trained on byte-frequency histograms to try to recover a XOR key without bruteforcing. (XOR is vulnerable to frequency analysis)

## Dataset

`data.csv` — 12,800 rows, 263 columns. Regenerate (or resize) it with:

```
python generate_data.py --per-key 50 --seed 1337
```

Each row is one ciphertext = plaintext XOR single-byte key.

| column | meaning |
| --- | --- |
| `sample_id` | row id (rows are shuffled) |
| `key` | **label** — the key byte, 0–255 |
| `plaintext_type` | english, source_code, json, log, url_list, base64, hexdump |
| `length` | ciphertext length in bytes |
| `length_bucket` | tiny (16–63), short (64–255), medium (256–1023), long (1024–4096) |
| `split` | train / val / test — 68/16/16, stratified so all 256 keys appear in each |
| `ciphertext_head_hex` | first 16 ciphertext bytes, hex (for sanity checks, not a feature) |
| `freq_000` … `freq_255` | **features** — normalised byte histogram, sums to 1.0 |

50 samples per key × 256 keys, so the classes are perfectly balanced. Plaintext
types and length buckets are mixed within every key, and every plaintext is
generated independently — no plaintext is shared between train and test.

## Models

**Direct classifier** — softmax regression on the 256 ciphertext frequency
columns, 256 key classes: **0.938 test accuracy** (tiny 0.87, short 0.94,
medium 0.97, long 0.97).

**Candidate scorer** (`python train_scorer.py`) — a 256-128-64-1 MLP that never
sees a key. It answers one question: *is this histogram real plaintext or
garbage?* To recover a key we decrypt under all 256 candidates and take the
highest-scoring one. **0.993 test accuracy** (0.998 top-3).

The scorer wins because it learns one concept instead of 256, and it gets its
training data free: XOR-ing by `k` permutes the histogram, so decrypting a
histogram with candidate `k` is just reordering the columns by `i ^ k`. Every
row yields 1 real plaintext histogram and 255 garbage ones — 8,704 ciphertexts
become a 60,928-example training pool.

|  | scorer | printable-mass rule |
| --- | --- | --- |
| long / medium / short | 1.000 | 0.40 / 0.25 / 0.15 |
| tiny (16–63 bytes) | 0.970 | 0.042 |
| base64 plaintext | 0.929 | 0.048 |

Remaining error is concentrated in tiny ciphertexts and base64 plaintext, which
is the expected place for it — base64 is close to uniform over its 64-character
alphabet, so there is little frequency structure to find.

Note: this is single-byte XOR. For a repeating multi-byte key, split the
ciphertext into columns by key position first, then apply this classifier per
column.
