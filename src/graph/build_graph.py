"""Assemble PyG HeteroData graph from concepts, relations, and metadata.

Outputs:
  data/graph/bibliograph.pt       — the HeteroData object
  data/graph/node_lookup.json     — int index → human-readable metadata per node type
  data/graph/evidence_lookup.json — pair key → evidence text from Ollama
"""
import json
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from src.config import (
    DATA_PROCESSED,
    DATA_GRAPH,
    GRAPH_SAVE_PATH,
    EMBEDDING_MODEL,
    GUTENBERG_BOOKS,
    RELATION_TYPES,
)


def _book_id_slug(book: dict) -> str:
    return book["title"].lower().replace(" ", "_").replace(",", "").replace("'", "")


def build_graph() -> HeteroData:
    concepts_path = DATA_PROCESSED / "concepts.json"
    relations_path = DATA_PROCESSED / "relations.jsonl"
    metadata_path = DATA_PROCESSED / "metadata.json"

    all_concepts: dict = json.loads(concepts_path.read_text())
    metadata: dict = json.loads(metadata_path.read_text())

    # load relations — deduplicate to one record per (source, target) pair
    relations: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    if relations_path.exists():
        with relations_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pair = (rec["source"], rec["target"])
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    relations.append(rec)

    encoder = SentenceTransformer(EMBEDDING_MODEL)

    # ── build node index ──────────────────────────────────────────────────────
    # concepts: global (not per-book) — shared concepts serve as bridges
    concept_to_idx: dict[str, int] = {}
    concept_to_books: dict[str, list[str]] = defaultdict(list)
    book_to_idx: dict[str, int] = {}
    author_to_idx: dict[str, int] = {}

    for book in GUTENBERG_BOOKS:
        bid = str(book["id"])
        meta = metadata.get(bid, {})
        title = meta.get("title", book["title"])
        author = meta.get("author", book["author"])
        slug = _book_id_slug(book)

        if title not in book_to_idx:
            book_to_idx[title] = len(book_to_idx)
        if author not in author_to_idx:
            author_to_idx[author] = len(author_to_idx)

        for concept in all_concepts.get(bid, []):
            if concept not in concept_to_idx:
                concept_to_idx[concept] = len(concept_to_idx)
            concept_to_books[concept].append(slug)

    # ── embeddings ────────────────────────────────────────────────────────────
    concept_texts = list(concept_to_idx.keys())
    book_texts = list(book_to_idx.keys())
    author_texts = list(author_to_idx.keys())

    print(f"  encoding {len(concept_texts)} concepts …")
    concept_emb = torch.tensor(encoder.encode(concept_texts, show_progress_bar=False), dtype=torch.float)
    book_emb = torch.tensor(encoder.encode(book_texts, show_progress_bar=False), dtype=torch.float)
    author_emb = torch.tensor(encoder.encode(author_texts, show_progress_bar=False), dtype=torch.float)

    data = HeteroData()
    data["concept"].x = concept_emb
    data["book"].x = book_emb
    data["author"].x = author_emb

    # ── semantic concept–concept edges ───────────────────────────────────────
    rel_edges: dict[str, tuple[list, list]] = {r: ([], []) for r in RELATION_TYPES}
    evidence_lookup: dict[str, dict] = {}

    for rec in relations:
        src_idx = concept_to_idx.get(rec["source"])
        tgt_idx = concept_to_idx.get(rec["target"])
        rel = rec.get("relation", "related_to")
        if src_idx is None or tgt_idx is None or rel not in rel_edges:
            continue
        rel_edges[rel][0].append(src_idx)
        rel_edges[rel][1].append(tgt_idx)
        # symmetric relations get both directions for message passing
        if rel in ("contradicts", "related_to"):
            rel_edges[rel][0].append(tgt_idx)
            rel_edges[rel][1].append(src_idx)
        pair_key = f"{rec['source']}||{rec['target']}"
        evidence_lookup[pair_key] = {
            "evidence": rec.get("evidence", ""),
            "confidence": rec.get("confidence", 0.5),
            "book_id": rec.get("book_id", ""),
        }

    for rel, (srcs, tgts) in rel_edges.items():
        if srcs:
            data["concept", rel, "concept"].edge_index = torch.tensor([srcs, tgts], dtype=torch.long)

    # ── structural edges ──────────────────────────────────────────────────────
    appears_in_src, appears_in_tgt = [], []
    has_concept_src, has_concept_tgt = [], []

    for book in GUTENBERG_BOOKS:
        bid = str(book["id"])
        meta = metadata.get(bid, {})
        title = meta.get("title", book["title"])
        b_idx = book_to_idx[title]

        for concept in all_concepts.get(bid, []):
            c_idx = concept_to_idx.get(concept)
            if c_idx is not None:
                appears_in_src.append(c_idx)
                appears_in_tgt.append(b_idx)
                has_concept_src.append(b_idx)
                has_concept_tgt.append(c_idx)

    data["concept", "appears_in", "book"].edge_index = torch.tensor(
        [appears_in_src, appears_in_tgt], dtype=torch.long
    )
    data["book", "has_concept", "concept"].edge_index = torch.tensor(
        [has_concept_src, has_concept_tgt], dtype=torch.long
    )

    written_by_src, written_by_tgt = [], []
    wrote_src, wrote_tgt = [], []

    for book in GUTENBERG_BOOKS:
        bid = str(book["id"])
        meta = metadata.get(bid, {})
        title = meta.get("title", book["title"])
        author = meta.get("author", book["author"])
        b_idx = book_to_idx[title]
        a_idx = author_to_idx[author]
        written_by_src.append(b_idx)
        written_by_tgt.append(a_idx)
        wrote_src.append(a_idx)
        wrote_tgt.append(b_idx)

    data["book", "written_by", "author"].edge_index = torch.tensor(
        [written_by_src, written_by_tgt], dtype=torch.long
    )
    data["author", "wrote", "book"].edge_index = torch.tensor(
        [wrote_src, wrote_tgt], dtype=torch.long
    )

    # ── save graph ────────────────────────────────────────────────────────────
    DATA_GRAPH.mkdir(parents=True, exist_ok=True)
    torch.save(data, GRAPH_SAVE_PATH)
    print(f"  graph saved → {GRAPH_SAVE_PATH}")

    # ── node lookup sidecar ───────────────────────────────────────────────────
    node_lookup = {
        "concept": {str(idx): {"concept_id": c, "canonical_name": c, "source_books": concept_to_books[c]}
                    for c, idx in concept_to_idx.items()},
        "book": {str(idx): {"title": t} for t, idx in book_to_idx.items()},
        "author": {str(idx): {"name": a} for a, idx in author_to_idx.items()},
    }
    lookup_path = DATA_GRAPH / "node_lookup.json"
    lookup_path.write_text(json.dumps(node_lookup, indent=2), encoding="utf-8")
    print(f"  node lookup saved → {lookup_path}")

    evidence_path = DATA_GRAPH / "evidence_lookup.json"
    evidence_path.write_text(json.dumps(evidence_lookup, indent=2), encoding="utf-8")
    print(f"  evidence lookup saved → {evidence_path}")

    return data


if __name__ == "__main__":
    build_graph()
