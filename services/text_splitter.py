def split_text(text: str, chunk_size: int = 300, overlap: int = 30):
    """Split text into chunks of a specified size with overlap."""

    if chunk_size < overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
