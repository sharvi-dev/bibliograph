"""API route handlers."""
from fastapi import APIRouter, HTTPException
from src.api.schemas import BridgesResponse, GraphResponse, ExplainRequest, ExplainResponse

router = APIRouter()


@router.get("/bridges", response_model=BridgesResponse)
async def get_bridges(book_a: str, book_b: str, top_k: int = 10):
    # TODO: load model + graph, score cross-book pairs, return top_k
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/graph", response_model=GraphResponse)
async def get_graph(book: str):
    # TODO: load graph, filter to subgraph for requested book, return nodes + edges
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/explain", response_model=ExplainResponse)
async def explain_bridge(body: ExplainRequest):
    # TODO: call Gemini API to generate one-sentence explanation
    raise HTTPException(status_code=501, detail="Not implemented yet")
