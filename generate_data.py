"""Build the training/testing dataset for xor_keyrec_ml.

Each row is one ciphertext produced by XOR-ing a plaintext with a single-byte
key. The features are the normalised byte-frequency histogram of that
ciphertext (256 bins); the label is the key byte that produced it.

Usage:
    python generate_data.py                  # writes data.csv
    python generate_data.py --per-key 100    # bigger set
"""

import argparse
import base64
import csv
import random

# --------------------------------------------------------------------------
# plaintext generators -- each returns roughly n characters of text
# --------------------------------------------------------------------------

COMMON = (
    "the be to of and a in that have i it for not on with he as you do at this "
    "but his by from they we say her she or an will my one all would there their "
    "what so up out if about who get which go me when make can like time no just "
    "him know take people into year your good some could them see other than then "
    "now look only come its over think also back after use two how our work first "
    "well way even new want because any these give day most us"
).split()

TAIL = (
    "system network memory buffer random analysis frequency distribution cipher "
    "message plaintext language english letters counting probability entropy "
    "training testing accuracy machine learning model feature vector histogram "
    "morning weather garden window coffee mountain river forest quiet evening "
    "children teacher student library question answer picture colour music "
    "market village island summer winter travel journey letter number street "
    "family friend office kitchen animal flower bridge castle harbour reason "
    "problem simple difficult possible important different available example "
    "development government community experience information environment "
    "considered following remember thousand hundred million science"
).split()

WORDS = COMMON * 6 + TAIL

IDENTS = (
    "count total value index buffer result data key freq hist item node cache "
    "config payload offset length window sample label score batch tokens path"
).split()

TYPES = ["int", "str", "float", "bool", "bytes", "list", "dict"]
HOSTS = ["example.com", "api.service.io", "cdn.static.net", "docs.site.org",
         "files.internal.local", "mail.provider.com", "shop.retail.co.uk"]
LEVELS = ["INFO", "WARN", "ERROR", "DEBUG", "TRACE"]
NAMES = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi"]


def gen_english(rng, n):
    out = []
    size = 0
    while size < n:
        words = [rng.choice(WORDS) for _ in range(rng.randint(4, 18))]
        words[0] = words[0].capitalize()
        if rng.random() < 0.08:
            words.insert(rng.randrange(len(words)), str(rng.randint(1, 9999)))
        sent = " ".join(words) + rng.choice(".....?!;,")
        sep = "\n\n" if rng.random() < 0.06 else " "
        out.append(sent + sep)
        size += len(sent) + len(sep)
    return "".join(out)


def gen_code(rng, n):
    out = []
    size = 0
    while size < n:
        a, b, c = (rng.choice(IDENTS) for _ in range(3))
        line = rng.choice([
            "def {}_{}({}, {}=None):".format(a, b, c, rng.choice(IDENTS)),
            "    {} = {}({}) + {}".format(
                a, rng.choice(["len", "int", "sum", "abs"]), b, rng.randint(0, 64)),
            "    if {} is not None and {} > {}:".format(a, b, rng.randint(0, 255)),
            "        return {{{!r}: {}, {!r}: {}}}".format(a, b, c, rng.randint(0, 999)),
            "    for {} in range({}):".format(a, rng.randint(2, 256)),
            "        {}[{}] += 1".format(a, b),
            "    # {} the {} before {}".format(rng.choice(WORDS), b, rng.choice(WORDS)),
            "{}: {} = {}".format(a, rng.choice(TYPES), rng.randint(0, 4096)),
            "    raise ValueError(msg.format({}, {}))".format(a, b),
            "",
        ])
        out.append(line + "\n")
        size += len(line) + 1
    return "".join(out)


def gen_json(rng, n):
    out = []
    size = 0
    while size < n:
        rec = ('{{"id": {}, {!r}: {!r}, {!r}: {:.3f}, "active": {}, '
               '"tags": [{!r}, {!r}]}}').format(
            rng.randint(1, 10 ** 6), rng.choice(IDENTS), rng.choice(NAMES),
            rng.choice(IDENTS), rng.uniform(0, 100),
            rng.choice(["true", "false"]),
            rng.choice(WORDS), rng.choice(WORDS))
        rec = rec.replace("'", '"')
        out.append(rec + ",\n")
        size += len(rec) + 2
    return "[" + "".join(out)[:-2] + "]"


def gen_log(rng, n):
    out = []
    size = 0
    while size < n:
        line = "2026-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}.{:03d}Z {:<5} {}.{}.{}.{} {} {}={}".format(
            rng.randint(1, 12), rng.randint(1, 28), rng.randint(0, 23),
            rng.randint(0, 59), rng.randint(0, 59), rng.randint(0, 999),
            rng.choice(LEVELS), rng.randint(1, 254), rng.randint(0, 255),
            rng.randint(0, 255), rng.randint(1, 254),
            " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 7))),
            rng.choice(IDENTS), rng.randint(0, 99999))
        out.append(line + "\n")
        size += len(line) + 1
    return "".join(out)


def gen_url(rng, n):
    out = []
    size = 0
    while size < n:
        line = "https://{}/{}/{}?{}={}&{}={}".format(
            rng.choice(HOSTS), rng.choice(WORDS), rng.choice(WORDS),
            rng.choice(IDENTS), rng.randint(1, 9999),
            rng.choice(IDENTS), rng.choice(NAMES))
        out.append(line + "\n")
        size += len(line) + 1
    return "".join(out)


def gen_base64(rng, n):
    raw = bytes(rng.randrange(256) for _ in range(int(n * 0.78) + 8))
    return base64.b64encode(raw).decode("ascii")


def gen_hexdump(rng, n):
    out = []
    size = 0
    while size < n:
        line = "{:08x}  {}".format(
            size, " ".join("{:02x}".format(rng.randrange(256)) for _ in range(16)))
        out.append(line + "\n")
        size += len(line) + 1
    return "".join(out)


GENERATORS = [
    ("english", gen_english, 0.40),
    ("source_code", gen_code, 0.14),
    ("json", gen_json, 0.12),
    ("log", gen_log, 0.12),
    ("url_list", gen_url, 0.08),
    ("base64", gen_base64, 0.08),
    ("hexdump", gen_hexdump, 0.06),
]

TYPE_NAMES = [g[0] for g in GENERATORS]
TYPE_WEIGHTS = [g[2] for g in GENERATORS]
TYPE_FUNCS = dict((g[0], g[1]) for g in GENERATORS)

# (name, min_bytes, max_bytes) -- short samples are the hard cases
BUCKETS = [
    ("tiny", 16, 63),
    ("short", 64, 255),
    ("medium", 256, 1023),
    ("long", 1024, 4096),
]


def make_sample(rng):
    """Return (plaintext_type, length_bucket, plaintext_bytes)."""
    name = rng.choices(TYPE_NAMES, weights=TYPE_WEIGHTS, k=1)[0]
    bucket, lo, hi = rng.choice(BUCKETS)
    length = rng.randint(lo, hi)
    data = TYPE_FUNCS[name](rng, length).encode("utf-8", "ignore")
    if len(data) < length:  # generator undershot: repeat to fill
        data = (data * (length // max(len(data), 1) + 1))
    return name, bucket, data[:length]


def histogram(data):
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    total = float(len(data))
    return [c / total for c in counts]


def fmt(v):
    return "0" if v == 0.0 else "{:.6f}".format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data.csv")
    ap.add_argument("--per-key", type=int, default=50,
                    help="samples per key byte (256 keys)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    header = (["sample_id", "key", "plaintext_type", "length", "length_bucket",
               "split", "ciphertext_head_hex"]
              + ["freq_{:03d}".format(i) for i in range(256)])

    rows = []
    for key in range(256):
        per_key = []
        for _ in range(args.per_key):
            ptype, bucket, plain = make_sample(rng)
            per_key.append((ptype, bucket, bytes(b ^ key for b in plain)))

        # stratified split: every key byte shows up in train, val and test
        rng.shuffle(per_key)
        n = len(per_key)
        n_train = int(round(n * 0.68))
        n_val = int(round(n * 0.16))
        splits = (["train"] * n_train + ["val"] * n_val
                  + ["test"] * (n - n_train - n_val))

        for (ptype, bucket, cipher), split in zip(per_key, splits):
            rows.append([0, key, ptype, len(cipher), bucket, split,
                         cipher[:16].hex()] + [fmt(v) for v in histogram(cipher)])

    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row[0] = i

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)

    print("wrote {}: {} rows x {} columns".format(args.out, len(rows), len(header)))


if __name__ == "__main__":
    main()
