"""API route handlers for BiblioGraph."""
import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    Bridge, BridgesResponse,
    GraphNode, GraphEdge, GraphResponse,
    ExplainRequest, ExplainResponse,
)
import src.api.state as state
from src.config import GUTENBERG_BOOKS

router = APIRouter()

BATCH = 4096

# slug ↔ title helpers
_SLUG_TO_TITLE: dict[str, str] = {}
_TITLE_TO_SLUG: dict[str, str] = {}
for _b in GUTENBERG_BOOKS:
    _slug = _b["title"].lower().replace(" ", "_").replace(",", "").replace("'", "")
    _SLUG_TO_TITLE[_slug] = _b["title"]
    _TITLE_TO_SLUG[_b["title"].lower()] = _slug

RELATION_COLORS = {
    "causes":       "#e74c3c",
    "contradicts":  "#8e44ad",
    "exemplifies":  "#27ae60",
    "extends":      "#2980b9",
    "is_a":         "#f39c12",
    "related_to":   "#95a5a6",
    "appears_in":   "#bdc3c7",
}


def _resolve_slug(book: str) -> str:
    """Accept either a slug or a full title and return the canonical slug."""
    if book in _SLUG_TO_TITLE:
        return book
    candidate = _TITLE_TO_SLUG.get(book.lower())
    if candidate:
        return candidate
    raise HTTPException(status_code=404, detail=f"Unknown book: '{book}'. Use a slug like 'the_republic'.")


# ── /bridges ─────────────────────────────────────────────────────────────────

@router.get("/bridges", response_model=BridgesResponse)
async def get_bridges(book_a: str, book_b: str, top_k: int = 10, exclusive: bool = True):
    """
    exclusive=True (default): only return bridges where concept_a appears
    solely in book_a and concept_b solely in book_b — avoids hub concepts
    that span many books dominating the results.
    """
    slug_a = _resolve_slug(book_a)
    slug_b = _resolve_slug(book_b)
    if slug_a == slug_b:
        raise HTTPException(status_code=400, detail="book_a and book_b must be different.")

    z_dict      = state.get("z_dict")
    cb          = state.get("cross_book_pairs")
    nl          = state.get("node_lookup")
    ev          = state.get("evidence_lookup")
    concepts    = nl["concept"]

    cb_ei = cb["edge_index"]

    # filter to pairs where one side is in slug_a and the other in slug_b
    mask = []
    for k in range(cb_ei.shape[1]):
        i, j = int(cb_ei[0, k]), int(cb_ei[1, k])
        bi = set(concepts[str(i)]["source_books"])
        bj = set(concepts[str(j)]["source_books"])
        ab = (slug_a in bi) and (slug_b in bj)
        ba = (slug_b in bi) and (slug_a in bj)
        if not (ab or ba):
            continue
        if exclusive:
            # require each concept to belong to only one book
            if ab and (len(bi) != 1 or len(bj) != 1):
                continue
            if ba and (len(bi) != 1 or len(bj) != 1):
                continue
        mask.append(k)

    if not mask:
        return BridgesResponse(bridges=[], book_a_slug=slug_a, book_b_slug=slug_b)

    idx_tensor = torch.tensor(mask, dtype=torch.long)
    filtered_ei = cb_ei[:, idx_tensor]

    # score in batches
    model = state.get("model")
    all_scores = []
    with torch.no_grad():
        for start in range(0, filtered_ei.shape[1], BATCH):
            batch = filtered_ei[:, start: start + BATCH]
            s = torch.sigmoid(model.predict_link(z_dict, batch))
            all_scores.append(s)
    scores_np = torch.cat(all_scores).numpy()

    top_local = np.argsort(scores_np)[::-1][:top_k]

    bridges = []
    for k in top_local:
        i, j = int(filtered_ei[0, k]), int(filtered_ei[1, k])
        ci, cj = concepts[str(i)], concepts[str(j)]
        name_a, name_b = ci["canonical_name"], cj["canonical_name"]
        ev_rec = ev.get(f"{name_a}||{name_b}", ev.get(f"{name_b}||{name_a}", {}))
        bridges.append(Bridge(
            concept_a=name_a,
            concept_b=name_b,
            books_a=ci["source_books"],
            books_b=cj["source_books"],
            score=float(scores_np[k]),
            evidence=ev_rec.get("evidence", ""),
            relation=ev_rec.get("relation", ""),
        ))

    return BridgesResponse(bridges=bridges, book_a_slug=slug_a, book_b_slug=slug_b)


# ── /graph ────────────────────────────────────────────────────────────────────

@router.get("/graph", response_model=GraphResponse)
async def get_graph(book: str):
    slug = _resolve_slug(book)

    b2c     = state.get("book_to_concepts")   # slug → {idx: name}
    nl      = state.get("node_lookup")
    data    = state.get("data")

    book_concepts = b2c.get(slug)
    if not book_concepts:
        raise HTTPException(status_code=404, detail=f"No concepts found for book '{slug}'.")

    concept_idx_set = set(book_concepts.keys())

    # concept nodes
    nodes: list[GraphNode] = [
        GraphNode(id=f"c{idx}", label=name, node_type="concept", books=[slug])
        for idx, name in book_concepts.items()
    ]

    # book node
    book_nl = nl["book"]
    b_entry = next(
        ((bid, m) for bid, m in book_nl.items()
         if m["title"].lower().replace(" ", "_").replace(",", "").replace("'", "") == slug),
        None,
    )
    if b_entry:
        bid, bmeta = b_entry
        nodes.append(GraphNode(id=f"book_{bid}", label=bmeta["title"], node_type="book"))

    # semantic concept–concept edges within this book
    edges: list[GraphEdge] = []
    seen: set[tuple] = set()
    for src_t, rel, dst_t in data.edge_types:
        if src_t != "concept" or dst_t != "concept":
            continue
        ei = data[src_t, rel, dst_t].edge_index
        for k in range(ei.shape[1]):
            s, d = int(ei[0, k]), int(ei[1, k])
            if s in concept_idx_set and d in concept_idx_set:
                canonical = (min(s, d), max(s, d), rel)
                if canonical not in seen:
                    seen.add(canonical)
                    edges.append(GraphEdge(source=f"c{s}", target=f"c{d}", relation=rel))

    return GraphResponse(nodes=nodes, edges=edges)


# ── /explain ─────────────────────────────────────────────────────────────────

@router.post("/explain", response_model=ExplainResponse)
async def explain_bridge(body: ExplainRequest):
    title_a = _SLUG_TO_TITLE.get(body.books_a[0], body.books_a[0]) if body.books_a else "unknown"
    title_b = _SLUG_TO_TITLE.get(body.books_b[0], body.books_b[0]) if body.books_b else "unknown"

    evidence_clause = ""
    if body.evidence:
        snippet = body.evidence[:200].rstrip()
        evidence_clause = f' In {title_a}, the concept appears in the context: "{snippet}..."'

    explanation = (
        f"BiblioGraph detected a latent conceptual bridge between '{body.concept_a}' "
        f"({title_a}) and '{body.concept_b}' ({title_b}) "
        f"with a confidence score of {body.score:.3f}.{evidence_clause} "
        f"Though these ideas never appear in the same text, the Graph Transformer "
        f"found that their embeddings cluster together in the shared concept space, "
        f"suggesting a deep structural resonance across these works."
    )
    return ExplainResponse(explanation=explanation)


# ── /books ────────────────────────────────────────────────────────────────────

@router.get("/books")
async def list_books():
    return [
        {"title": _SLUG_TO_TITLE[slug], "slug": slug}
        for slug in _SLUG_TO_TITLE
    ]
