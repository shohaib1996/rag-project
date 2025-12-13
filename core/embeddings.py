from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert list of texts into embeddings.
    """
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
