from fastapi import FastAPI, HTTPException

from app.schemas import AskRequest, AskResponse
from app.graph import answer_question

app = FastAPI(title="Legixo GenAI Take-home — Document Q&A API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = answer_question(question)
    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        trace={"steps": result["trace"]},
    )