from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("~/Desktop/")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# 1. LOADER
# ============================================================

def load_documents(data_dir):
    """
    Load all .txt files from a directory.

    Returns:
        list[dict]:
            {
                "filename": str,
                "text": str
            }
    """

    documents = []

    for file_path in sorted(data_dir.glob("*.txt")):

        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append({
            "filename": file_path.name,
            "text": text
        })

    return documents


# ============================================================
# 2. CHUNKER
# ============================================================

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks.

    Example:

        chunk 1: characters 0   -> 500
        chunk 2: characters 400 -> 900
        chunk 3: characters 800 -> 1300

    The overlap helps preserve context between chunks.
    """

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    start = 0
    step = chunk_size - overlap

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def build_chunks(documents):
    """
    Chunk every document while preserving its metadata.
    """

    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        for chunk_id, chunk in enumerate(document_chunks):

            chunks.append({
                "filename": document["filename"],
                "chunk_id": chunk_id,
                "text": chunk
            })

    return chunks


# ============================================================
# 3. EMBEDDER
# ============================================================

class Embedder:

    def __init__(self, model_name=MODEL_NAME):

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Convert text into embedding vectors.

        Returns:
            numpy array of shape:

                (number_of_texts, embedding_dimension)
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings


# ============================================================
# 4. VECTOR INDEX
# ============================================================

class VectorIndex:

    def __init__(self, embeddings, chunks):

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must equal number of chunks"
            )

        self.embeddings = embeddings
        self.chunks = chunks

    def search(self, query_embedding, top_k=5):
        """
        Find the top-k most similar chunks.

        Because embeddings are normalized, the dot product
        is equivalent to cosine similarity.
        """

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:

            results.append({
                "filename": self.chunks[index]["filename"],
                "chunk_id": self.chunks[index]["chunk_id"],
                "text": self.chunks[index]["text"],
                "score": float(scores[index])
            })

        return results


# ============================================================
# BUILD THE INDEX
# ============================================================

def build_index():

    print("\nLoading documents...")

    documents = load_documents(DATA_DIR)

    print(f"Loaded {len(documents)} documents")

    if not documents:
        raise ValueError("No .txt files found in the data directory")

    print("\nChunking documents...")

    chunks = build_chunks(documents)

    print(f"Created {len(chunks)} chunks")

    print("\nGenerating embeddings...")

    embedder = Embedder()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedder.encode(texts)

    print(f"Embedding shape: {embeddings.shape}")

    print("\nBuilding vector index...")

    index = VectorIndex(
        embeddings=embeddings,
        chunks=chunks
    )

    print("Vector index ready!")

    return embedder, index


# ============================================================
# SEARCH
# ============================================================

def search(query, embedder, index, top_k=5):

    query_embedding = embedder.encode([query])[0]

    results = index.search(
        query_embedding=query_embedding,
        top_k=top_k
    )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    embedder, index = build_index()

    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH")
    print("=" * 60)

    while True:

        query = input("\nEnter your query (or 'quit'): ")

        if query.lower() == "quit":
            break

        results = search(
            query=query,
            embedder=embedder,
            index=index,
            top_k=5
        )

        print("\nResults:")

        for rank, result in enumerate(results, start=1):

            print("\n" + "-" * 60)

            print(f"Rank:     {rank}")
            print(f"File:     {result['filename']}")
            print(f"Chunk:    {result['chunk_id']}")
            print(f"Score:    {result['score']:.4f}")

            print("\nText:")
            print(result["text"])