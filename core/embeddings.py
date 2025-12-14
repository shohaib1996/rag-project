# from sentence_transformers import SentenceTransformer
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert list of texts into embeddings.
    """
    # embeddings = model.encode(texts, convert_to_numpy=True)
    # return embeddings.tolist()

    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [data.embedding for data in response.data]
