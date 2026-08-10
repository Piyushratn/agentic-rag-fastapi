# LangGraph layout

## Diagram

    ┌───────▶│  retrieve  │◀────────────┐
    │        └─────┬──────┘              │
    │               │                    │
    │               ▼                    │
    │        ┌──────────────┐            │
    │        │ grade_chunks │            │
    │        └─────┬────────┘            │
    │      good /  │  \ bad              │
    │      ┌───────┘   └────────┐        │
    │      ▼                    ▼        │
    │ ┌──────────┐   loops < max?        │
    │ │ generate │        │        │     │
    │ └────┬─────┘   yes  │  no    │     │
    │      │              ▼        ▼     │
    │      ▼         ┌──────────┐ ┌────────────────┐
    │    (END)       │ rewrite  │ │ cannot_answer  │
    │                └────┬─────┘ └───────┬────────┘
    │                     │               │
    └─────────────────────┘               ▼
                                         (END)


## Nodes

| Node | What it does |
|---|---|
| `retrieve` | Embeds the current search query (initially the raw question, later a rewritten query) using Gemini's `gemini-embedding-001`, and queries Pinecone for the top `TOP_K` chunks with metadata (`source_file`, `chunk_id`, `chunk_index`, `text`). |
| `grade_chunks` | **Branch point.** First does a cheap check: is the best cosine similarity score above `SCORE_THRESHOLD`? If yes, asks the LLM a strict yes/no question — "does this context actually answer the question?" — and sets `grade = "good"` or `"bad"`. (Named `grade_chunks` rather than `grade` because LangGraph does not allow a node name to collide with a state key name — `grade` is also a field in `GraphState`.) |
| `rewrite_query` | Only reached on the bad path. Asks the LLM to turn the original question into a short, keyword-focused search query, increments `loop_count`, then the graph loops back to `retrieve`. |
| `generate` | Only reached on the good path. Builds an answer using **only** the retrieved chunks, requiring an inline `[source_file#chunk_index]` citation on every factual sentence. Also builds the structured `citations` list returned by the API. |
| `cannot_answer` | Reached when the bad path has exhausted its retries. Returns a plain "not found in the documents" message with an empty citation list — no guessing. |

## Branch logic (`route_after_grade`)

```python
def route_after_grade(state):
    if state["grade"] == "good":
        return "generate"
    if state["loop_count"] < MAX_GRAPH_LOOPS:
        return "rewrite"
    return "cannot_answer"
```

## Loop cap (two layers, on purpose)

1. **Explicit counter** — `state["loop_count"]` is incremented in `rewrite_query_node` and checked in `route_after_grade` against `MAX_GRAPH_LOOPS` (default `2`). This is the real, intentional limit.
2. **Hard safety net** — `graph.invoke(..., config={"recursion_limit": 20})`. Even if the counter logic were ever broken by a future change, LangGraph itself will refuse to run past 20 super-steps and raise instead of looping forever.

## State shape

```python
class GraphState(TypedDict):
    question: str          # original user question, never mutated
    search_query: str       # what retrieve actually embeds/searches (starts = question)
    retrieved: list[dict]    # chunks from the last retrieve() call
    grade: str               # "good" | "bad"
    loop_count: int          # number of rewrite attempts so far
    answer: str               # final answer text
    citations: list[dict]      # structured citations returned to the API
    trace: list[str]            # human-readable step log, returned as `trace.steps`
```

## Why a graph instead of one big prompt

- The **branch** means the model can't just make something up when retrieval is weak — there's a real decision point with a different code path for "confident" vs "not confident."
- The **rewrite loop** gives the system one or two extra shots at finding the right chunk if the user's phrasing doesn't match the document's phrasing.
- The **loop cap** guarantees termination and a bounded, predictable cost per request.
- `trace.steps` in the API response makes every request auditable — you can see exactly which path was taken and why.

## Known limitations (honest notes)

- `citations` in the API response currently includes **all** chunks retrieved for the query, not only the ones actually referenced inline in the generated answer text. All returned chunks are real (never fabricated), but the list isn't filtered down to exactly what's cited. A future improvement would be to parse the `[source_file#chunk_index]` tags out of the generated answer and filter `citations` to just those.
- Embedding dimension is fixed at 768 (scaled down from Gemini's native 3072 via `output_dimensionality`) to keep Pinecone storage light for this small corpus.                                         