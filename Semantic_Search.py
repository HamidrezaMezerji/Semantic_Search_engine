

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("/Users/hamid/Documents/project/semantic_search/semantic-search-data")

TOP_K = 5

# Field labels in NORMALIZED form (ZWNJ -> space), because we match against
# clean_text() output.
FIELDS = [
    "شناسه",
    "دسته بندی",
    "عنوان",
    "وضعیت",
    "تاریخ",
    "شخص",
    "تامین کننده",
    "واحد درخواست دهنده",
    "توضیحات",
    "خلاصه درخواست",
    "اقلام",
    "مبلغ",
    "روش پرداخت",
    "ارز",
    "گروه کالا",
    "مرجع",
    "بانک",
]

# Only these fields contribute to the semantic representation.
SEMANTIC_FIELDS = [
    "عنوان",
    "توضیحات",
    "خلاصه درخواست",
    "اقلام",
]

# These fields remain metadata (shown but not embedded).
METADATA_FIELDS = [
    "دسته بندی",
    "وضعیت",
    "تاریخ",
    "شخص",
    "تامین کننده",
    "واحد درخواست دهنده",
    "شناسه",
    "مبلغ",
    "روش پرداخت",
    "ارز",
    "گروه کالا",
    "مرجع",
    "بانک",
]

# --- from-scratch model hyper-parameters -------------------
CHAR_NGRAM_SIZE = 3          # character n-gram length for the typo layer
WORD_WINDOW = 3              # co-occurrence context window (each side)
CONTEXT_TOP = 2000           # cap vocabulary/context size
PPMI_DOWNSAMPLE = 1e-4       # frequent-token downsampling (better vectors)
RARE_WORD_CUTOFF = 2         # drop words appearing fewer times than this
HYBRID_W = 0.75              # weight of word-vector signal (0.75) vs char (0.25)
MIN_SCORE = 0.20             # ignore results scoring below this (weak/noise)


# ============================================================
# PERSIAN NORMALIZATION
# ============================================================

def normalize_persian(text):
    """Normalize Persian/Arabic characters, digits and spacing."""

    replacements = str.maketrans({
        "ي": "ی", "ى": "ی", "ك": "ک",
        "أ": "ا", "إ": "ا", "آ": "ا",
        "\u200c": " ",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    })

    text = text.translate(replacements)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text):
    """Remove HTML and normalize text."""
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_persian(text)


def tokenize_units(text):
    """Split a line into loose tokens, splitting on '-', '|' too so  اقلام
    product / spec / quantity columns become individual matches."""
    text = text.replace("|", " ").replace("-", " ")
    return [t for t in text.split() if t]


# ============================================================
# PARSING
# ============================================================

def parse_record(text):
    """Extract known fields from a record."""

    text = clean_text(text)
    labels = "|".join(map(re.escape, FIELDS))
    pattern = rf"({labels}):\s*(.*?)(?=\s+(?:{labels}):|\s*---|\Z)"
    return {
        field: value.strip()
        for field, value in re.findall(pattern, text, re.DOTALL)
    }


def load_documents(data_dir):
    """Load and parse all TXT files."""

    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    documents = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        documents.append({
            "filename": file.name,
            "fields": parse_record(text),
        })
    return documents


def create_semantic_text(document):
    """Only semantic content, concatenated exactly as it will be indexed."""
    fields = document["fields"]
    parts = []
    for field in SEMANTIC_FIELDS:
        value = fields.get(field)
        if not value:
            continue
        parts.append(value)
    return " ".join(parts)


def sem_tokens(document):
    """Token stream over the semantic text of a document."""
    return tokenize_units(create_semantic_text(document))


# ============================================================
# FROM-SCRATCH DISTRIBUTIONAL MODEL
# ============================================================

class SemanticModel:
    """
    A corpus-trained semantic model built without any pretrained weights.

    Two signal families:

      * word vectors  : rows of a PPMI-weighted word x word co-occurrence
                        matrix. Words that appear in similar local contexts
                        get similar vectors, giving genuine distributional
                        semantics.
      * char n-grams  : character 3-gram bag, robust to spelling noise.

    Document vectors are a TF-IDF weighted sum of their word vectors,
    concatenated with a TF-IDF weighted char-n-gram vector.
    """

    def __init__(self, n_gram=CHAR_NGRAM_SIZE, window=WORD_WINDOW,
                 pmi_downsample=PPMI_DOWNSAMPLE, rare_cutoff=RARE_WORD_CUTOFF):
        self.n_gram = n_gram
        self.window = window
        self.pmi_downsample = pmi_downsample
        self.rare_cutoff = rare_cutoff
        self.word_vecs = None
        self.w2i = {}
        self.idf_word = None
        self.ngrams = None
        self.ngram_i = {}
        self.idf_ngram = None
        self.dim = 0

    # -------------------------------------------------- training

    def _build_vocab(self, word_lists):
        counts = Counter()
        for words in word_lists:
            for w in words:
                counts[w] += 1
        vocab = {w for w, c in counts.items()
                 if c >= self.rare_cutoff}
        return counts, vocab

    def fit(self, documents):
        """Train the model on the corpus and build the document matrix."""
        word_lists = [sem_tokens(d) for d in documents]

        # ---- 1. word vectors (PPMI co-occurrence) -------------
        counts, vocab = self._build_vocab(word_lists)

        # downsampled token stream for context building
        stream = []
        for words in word_lists:
            for w in words:
                if w not in vocab:
                    continue
                f = counts[w]
                keep = 1.0 - math.sqrt(self.pmi_downsample / (f / max(1, sum(counts.values()))))
                if random_keep(keep):
                    stream.append(w)

        co = defaultdict(Counter)      # word -> Counter(context word)
        for i, w in enumerate(stream):
            for j in range(max(0, i - self.window), min(len(stream), i + self.window + 1)):
                if i == j:
                    continue
                co[w][stream[j]] += 1

        # cap context vocab
        ctx_counts = Counter()
        for cc in co.values():
            ctx_counts.update(cc)
        ctx_vocab = [w for w, _ in ctx_counts.most_common(CONTEXT_TOP)]

        ctx_i = {w: i for i, w in enumerate(ctx_vocab)}
        word_vocab = [w for w in vocab if w in co]
        self.w2i = {w: i for i, w in enumerate(word_vocab)}

        total = sum(sum(cc.values()) for cc in co.values())
        if total == 0:
            raise RuntimeError("empty co-occurrence matrix")

        V = len(word_vocab)
        D = len(ctx_vocab)
        mat = np.zeros((V, D))
        row_total = {}
        for w in word_vocab:
            cc = co[w]
            row_total[w] = sum(cc.values())
            for c, cnt in cc.items():
                if c in ctx_i:
                    mat[self.w2i[w], ctx_i[c]] = cnt

        # PPMI: log( P(w,c) / (P(w)*P(c)) ) clipped at 0
        col_total = mat.sum(axis=0)
        p_pair = mat / total
        p_word = (np.array([row_total[w] for w in word_vocab], dtype=float) / total).reshape(-1, 1)
        p_ctx = (col_total / total).reshape(1, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(p_pair / (p_word @ p_ctx * max(1e-12, 1.0)))
            ppmi = np.maximum(pmi, 0.0)

        # squash + length-normalise rows -> word vectors
        wvec = np.sqrt(ppmi)
        norms = np.linalg.norm(wvec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.word_vecs = normalize_m(wvec / norms)
        self.dim = D

        # ---- 2. word IDF for weighting document sums ----------
        df_word = Counter()
        for doc_words in word_lists:
            df_word.update(set(doc_words))
        self.idf_word = {
            w: math.log((len(documents) + 1) / (df_word.get(w, 0) + 1)) + 1.0
            for w in word_vocab
        }

        # ---- 3. character n-grams ------------------------------
        ngram_df = Counter()
        doc_ngrams = []
        for doc_words in word_lists:
            ngs = self._char_ngrams(doc_words)
            doc_ngrams.append(ngs)
            ngram_df.update(set(ngs))

        self.ngrams = sorted(ngram_df)
        self.ngram_i = {g: i for i, g in enumerate(self.ngrams)}
        self.idf_ngram = {
            g: math.log((len(documents) + 1) / (ngram_df.get(g, 0) + 1)) + 1.0
            for g in self.ngrams
        }

        return self._embed_documents(word_lists, doc_ngrams)

    def _char_ngrams(self, words):
        text = " ".join(words)
        n = self.n_gram
        pad = " " * (n - 1)
        text = pad + text + pad
        return [text[i:i + n] for i in range(len(text) - n + 1)]

    # -------------------------------------------------- embedding
    #
    # Each document/query is represented by TWO unit-normalized vectors kept
    # separately (never concatenated into one vector), which lets search()
    # blend the two signals with the HYBRID_W weight and keeps each cosine
    # score well-formed (<= 1).

    def _embed_documents(self, word_lists, doc_ngrams):
        wm = np.zeros((len(word_lists), self.dim))
        gn = np.zeros((len(word_lists), len(self.ngrams)))
        for i, (words, ngs) in enumerate(zip(word_lists, doc_ngrams)):
            wm[i], gn[i] = self.embed_parts(words, ngs)
        return normalize_m(wm), normalize_m(gn)

    def embed_doc(self, words, ngs=None):
        """Return (word_part, ngram_part) unit vectors for a token list."""
        return self.embed_parts(words, ngs)

    def embed_parts(self, words, ngs=None):
        wvec = np.zeros(self.dim, dtype=float)
        for w in words:
            i = self.w2i.get(w)
            if i is None:
                continue
            wvec += self.word_vecs[i] * self.idf_word.get(w, 1.0)

        if ngs is None:
            ngs = self._char_ngrams(words)
        gvec = np.zeros(len(self.ngrams), dtype=float)
        for g in ngs:
            j = self.ngram_i.get(g)
            if j is None:
                continue
            gvec[j] += self.idf_ngram.get(g, 1.0)

        return normalize_m(wvec.reshape(1, -1))[0], normalize_m(gvec.reshape(1, -1))[0]

    def wdim(self):
        return self.dim

    def gdim(self):
        return len(self.ngrams) if self.ngrams else 0


def random_keep(p):
    import random
    return random.random() < p


def normalize_m(m):
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# ============================================================
# INDEX / SEARCH
# ============================================================

def build_index(documents, model):
    docs_words = [sem_tokens(d) for d in documents]
    ngrams = [model._char_ngrams(w) for w in docs_words]
    return model._embed_documents(docs_words, ngrams)


def search(query, documents, embeddings, model, top_k=TOP_K):
    """embeddings is the (word_matrix, ngram_matrix) pair from build_index."""
    q = normalize_persian(query)
    if not q:
        return []
    q_words = tokenize_units(q)
    qw, qg = model.embed_doc(q_words)

    wm, gm = embeddings
    w_scores = wm @ qw
    g_scores = gm @ qg
    scores = HYBRID_W * w_scores + (1.0 - HYBRID_W) * g_scores

    indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in indices:
        if float(scores[i]) < MIN_SCORE:
            break
        results.append({
            "score": float(scores[i]),
            "filename": documents[i]["filename"],
            "fields": documents[i]["fields"],
            "semantic_text": create_semantic_text(documents[i]),
        })
    return results


# ============================================================
# VALIDATION
# ============================================================

def build_engine(data_dir=DATA_DIR):
    """Load docs, train the from-scratch model, build the index."""
    documents = load_documents(data_dir)
    print(f"Loaded {len(documents)} documents.")
    model = SemanticModel()
    print("Training from-scratch semantic model on corpus...")
    embeddings = model.fit(documents)
    print(f"Index shape: words={embeddings[0].shape}, n-grams={embeddings[1].shape}")
    return documents, embeddings, model


def validate_documents(documents):
    total = len(documents)
    coverage = {
        field: sum(field in doc["fields"] for doc in documents)
        for field in FIELDS
    }
    missing = {
        doc["filename"]: [f for f in FIELDS if f not in doc["fields"]]
        for doc in documents if any(f not in doc["fields"] for f in FIELDS)
    }
    ids = [d["fields"]["شناسه"] for d in documents if d["fields"].get("شناسه")]
    dup = {rid for rid in ids if ids.count(rid) > 1}

    print("\n" + "=" * 65)
    print("PARSING VALIDATION REPORT")
    print("=" * 65)
    print(f"\nDocuments: {total}")
    print("\nFIELD COVERAGE")
    print("-" * 65)
    for field, count in coverage.items():
        print(f"{field:<25} {count:>3}/{total:<3} ({count/total*100:6.2f}%)")
    print("\nMISSING FIELDS")
    print("-" * 65)
    if not missing:
        print("None")
    else:
        for fn, fl in missing.items():
            print(f"{fn}: {', '.join(fl)}")
    print("\nDUPLICATE IDS")
    print("-" * 65)
    print("None" if not dup else ", ".join(dup))
    print("\nSTATUS")
    print("-" * 65)
    print("Parsing validation passed" if not missing and not dup
          else "Parsing validation completed with warnings")
    print("=" * 65)
    return not missing and not dup


# ============================================================
# DISPLAY (CLI)
# ============================================================

def display_results(query, results):
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    if not results:
        print("No results.")
        return
    for rank, result in enumerate(results, start=1):
        fields = result["fields"]
        print(f"\n#{rank} | Score: {result['score']:.4f}")
        print(f"File: {result['filename']}")
        print(f"Semantic content: {result['semantic_text']}")
        print("Metadata:")
        for field in METADATA_FIELDS:
            value = fields.get(field, "(ندارد)")
            if len(value) > 120:
                value = value[:120] + "..."
            print(f"  {field}: {value}")
        print("-" * 70)


# ============================================================
# CLI
# ============================================================

def interactive_search(documents, embeddings, model):
    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH (from scratch)")
    print("=" * 70)
    print(f"Indexed documents: {len(documents)}")
    print("Type a query or 'exit' to quit.")
    while True:
        query = input("\nSearch > ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not query:
            continue
        results = search(query, documents, embeddings, model)
        display_results(query, results)


def main():
    print("=" * 70)
    print("SEMANTIC SEARCH ENGINE (implemented from scratch)")
    print("=" * 70)

    documents, embeddings, model = build_engine()

    validate_documents(documents)
    print("\nSemantic fields: " + " + ".join(SEMANTIC_FIELDS))
    print("Metadata fields : " + " + ".join(METADATA_FIELDS))

    interactive_search(documents, embeddings, model)


if __name__ == "__main__":
    main()
