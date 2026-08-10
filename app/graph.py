import json
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, LLM_MODEL, TOP_K, SCORE_THRESHOLD, MAX_GRAPH_LOOPS, PINECONE_NAMESPACE
from app.embeddings import embed_query
from app.pinecone_client import get_index

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


class GraphState(TypedDict):
    question: str
    search_query: str
    retrieved: List[dict]
    grade: str
    loop_count: int
    answer: str
    citations: List[dict]
    trace: List[str]


def retrieve_node(state: GraphState) -> GraphState:
    query = state.get("search_query") or state["question"]
    vector = embed_query(query)
    index = get_index()
    result = index.query(
        vector=vector,
        top_k=TOP_K,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
    )

    retrieved = []
    for match in result.get("matches", []):
        md = match.get("metadata", {}) or {}
        retrieved.append(
            {
                "chunk_id": md.get("chunk_id"),
                "source_file": md.get("source_file"),
                "chunk_index": md.get("chunk_index"),
                "text": md.get("text", ""),
                "score": match.get("score", 0.0),
            }
        )

    state["retrieved"] = retrieved
    top_score = f"{retrieved[0]['score']:.3f}" if retrieved else "n/a"
    state.setdefault("trace", []).append(
        f"retrieve: query='{query}' -> {len(retrieved)} chunks (top score={top_score})"
    )
    return state


def grade_node(state: GraphState) -> GraphState:
    retrieved = state.get("retrieved", [])

    if not retrieved or retrieved[0]["score"] < SCORE_THRESHOLD:
        state["grade"] = "bad"
        state["trace"].append("grade: bad (below similarity threshold)")
        return state

    context = "\n\n".join(f"[{r['source_file']}#{r['chunk_index']}] {r['text']}" for r in retrieved)
    prompt = (
        "You are a strict grader. Decide if the CONTEXT below contains enough information "
        "to answer the QUESTION. Reply with strict JSON only, no other text: "
        '{"sufficient": true} or {"sufficient": false}.\n\n'
        f"QUESTION: {state['question']}\n\nCONTEXT:\n{context}"
    )
    resp = get_client().models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    )
    raw = (resp.text or "").strip()
    try:
        sufficient = bool(json.loads(raw).get("sufficient", False))
    except Exception:
        sufficient = "true" in raw.lower()

    state["grade"] = "good" if sufficient else "bad"
    state["trace"].append(f"grade: {'good' if sufficient else 'bad'} (LLM relevance check)")
    return state


def rewrite_query_node(state: GraphState) -> GraphState:
    state["loop_count"] = state.get("loop_count", 0) + 1
    prompt = (
        "Rewrite the QUESTION below into a short, keyword-focused search query that would "
        "retrieve better matching passages from a small set of legal-style notes "
        "(memos, agreements, hearing notices, statute excerpts, settlement notes, leases). "
        "Return ONLY the rewritten query, nothing else.\n\n"
        f"QUESTION: {state['question']}"
    )
    resp = get_client().models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    new_query = (resp.text or "").strip()
    state["search_query"] = new_query
    state["trace"].append(f"rewrite_query (attempt {state['loop_count']}): '{new_query}'")
    return state


def generate_node(state: GraphState) -> GraphState:
    retrieved = state["retrieved"]
    context = "\n\n".join(f"[{r['source_file']}#{r['chunk_index']}] {r['text']}" for r in retrieved)
    prompt = (
        "Answer the QUESTION using ONLY the CONTEXT below — do not use outside knowledge. "
        "Every factual sentence must end with a citation in the form [source_file#chunk_index]. "
        "If the context only partially answers the question, answer what you can and say "
        "plainly what is missing.\n\n"
        f"QUESTION: {state['question']}\n\nCONTEXT:\n{context}\n\nANSWER:"
    )
    resp = get_client().models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    state["answer"] = (resp.text or "").strip()
    state["citations"] = [
        {
            "source_file": r["source_file"],
            "chunk_id": r["chunk_id"],
            "chunk_index": r["chunk_index"],
            "text_snippet": (r["text"][:220] + "...") if len(r["text"]) > 220 else r["text"],
            "score": r["score"],
        }
        for r in retrieved
    ]
    state["trace"].append("generate: answer produced from retrieved context")
    return state


def cannot_answer_node(state: GraphState) -> GraphState:
    state["answer"] = (
        "I could not find this in the provided documents. "
        "Try rephrasing, or this may be outside the current corpus."
    )
    state["citations"] = []
    state["trace"].append("cannot_answer: retrieval loops exhausted without sufficient context")
    return state


def route_after_grade(state: GraphState) -> str:
    if state["grade"] == "good":
        return "generate"
    if state.get("loop_count", 0) < MAX_GRAPH_LOOPS:
        return "rewrite"
    return "cannot_answer"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_chunks", grade_node)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("generate", generate_node)
    graph.add_node("cannot_answer", cannot_answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_chunks")
    graph.add_conditional_edges(
        "grade_chunks",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "cannot_answer": "cannot_answer"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("cannot_answer", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def answer_question(question: str) -> dict:
    graph = get_graph()
    initial_state: GraphState = {
        "question": question,
        "search_query": question,
        "retrieved": [],
        "grade": "",
        "loop_count": 0,
        "answer": "",
        "citations": [],
        "trace": [],
    }
    final_state = graph.invoke(initial_state, config={"recursion_limit": 20})
    return {
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "trace": final_state["trace"],
    }