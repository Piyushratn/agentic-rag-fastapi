from pinecone import Pinecone, ServerlessSpec
from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIM,
)

_pc = None


def get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def ensure_index():
    pc = get_pinecone()
    existing_names = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_names:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(PINECONE_INDEX_NAME)


def get_index():
    pc = get_pinecone()
    return pc.Index(PINECONE_INDEX_NAME)