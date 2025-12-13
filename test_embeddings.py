from core.embeddings import embed_texts

texts = ["FastAPI is a modern web framework", "Embeddings convert text to vectors"]

vectors = embed_texts(texts)

print(len(vectors))
print(len(vectors[0]))
