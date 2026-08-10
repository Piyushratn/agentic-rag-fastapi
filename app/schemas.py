from typing import List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Citation(BaseModel):
    source_file: str
    chunk_id: str
    chunk_index: int
    text_snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    trace: Optional[dict] = None