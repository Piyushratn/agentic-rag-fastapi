from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input text, same order."""
    client = get_client()
    resp = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return [e.values for e in resp.embeddings]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]