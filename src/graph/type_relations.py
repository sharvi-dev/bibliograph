"""Classify relations between co-occurring concept pairs using keyword heuristics.

No API calls — purely local string matching on the context paragraph.
Output format is identical to the Gemini version so build_graph.py is unchanged.

Confidence scores:
  0.85  strong multi-word pattern match
  0.70  single keyword match
  0.55  fallback related_to
"""
import hashlib
import json
import random
from src.config import DATA_PROCESSED, RELATION_TYPES, GUTENBERG_BOOKS

MAX_CONCEPTS_USED = 100
MAX_CONCEPTS_PER_PARA = 6
MAX_EDGES_PER_BOOK = 300   # no API cost, so we can afford more edges

# keyword patterns ordered by specificity; checked in sequence, first match wins
_PATTERNS: list[tuple[str, list[str], float]] = [
    ("causes", [
        "causes", "leads to", "results in", "produces", "contributes to",
        "gives rise to", "brings about", "promotes", "drives", "increases",
        "decreases", "determines", "generates", "induces", "creates",
    ], 0.70),
    ("contradicts", [
        "contradicts", "opposes", "conflicts with", "incompatible", "contrary to",
        "negates", "denies", "inconsistent", "at odds", "in opposition",
        "unlike", "against", "refutes", "disputes",
    ], 0.70),
    ("is_a", [
        "is a type of", "is a form of", "is a kind of", "is a species of",
        "is an instance of", "is a subset of", "is a class of",
        "is a variety of", "is a mode of",
    ], 0.85),
    ("exemplifies", [
        "for example", "for instance", "such as", "is an example of",
        "illustrates", "demonstrates", "manifests", "is a case of",
        "serves as an example", "embodies",
    ], 0.70),
    ("extends", [
        "extends", "builds on", "builds upon", "expands", "elaborates",
        "develops", "continues", "advances", "refines", "adds to",
        "goes beyond", "further develops",
    ], 0.70),
]


def _pair_key(a: str, b: str) -> str:
    canonical = tuple(sorted([a, b]))
    return hashlib.md5("|".join(canonical).encode()).hexdigest()


def _classify_heuristic(concept_a: str, concept_b: str, context: str) -> dict:
    ctx = context.lower()
    for relation, keywords, confidence in _PATTERNS:
        if any(kw in ctx for kw in keywords):
            # find whichever keyword matched to use as evidence
            matched = next(kw for kw in keywords if kw in ctx)
            # extract a short snippet around the match for evidence
            idx = ctx.find(matched)
            snippet = context[max(0, idx - 40): idx + len(matched) + 40].strip()
            return {"relation": relation, "confidence": confidence, "evidence": snippet}

    return {
        "relation": "related_to",
        "confidence": 0.55,
        "evidence": f"'{concept_a}' and '{concept_b}' co-occur in the same passage.",
    }


def extract_relations_for_book(
    book_id: int, concepts: list[str], text: str, already_seen: set[str]
) -> list[dict]:
    concepts = concepts[:MAX_CONCEPTS_USED]
    concept_set = set(concepts)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    best: dict[str, dict] = {}
    edge_count = 0

    for para in paragraphs:
        if edge_count >= MAX_EDGES_PER_BOOK:
            break
        para_lower = para.lower()
        present = [c for c in concept_set if c in para_lower]
        if len(present) < 2:
            continue
        if len(present) > MAX_CONCEPTS_PER_PARA:
            present = random.sample(present, MAX_CONCEPTS_PER_PARA)

        for i, a in enumerate(present):
            for b in present[i + 1:]:
                if edge_count >= MAX_EDGES_PER_BOOK:
                    break
                key = _pair_key(a, b)
                if key in already_seen:
                    continue
                result = _classify_heuristic(a, b, para)
                if key not in best or result["confidence"] > best[key]["confidence"]:
                    best[key] = {
                        "source":     a,
                        "target":     b,
                        "relation":   result["relation"],
                        "confidence": result["confidence"],
                        "evidence":   result["evidence"],
                        "book_id":    str(book_id),
                    }
                if key not in already_seen:
                    edge_count += 1

    return list(best.values())


def type_all_relations() -> None:
    concepts_path = DATA_PROCESSED / "concepts.json"
    all_concepts: dict = json.loads(concepts_path.read_text())
    out_path = DATA_PROCESSED / "relations.jsonl"

    already_seen: set[str] = set()
    done_books: set[str] = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                already_seen.add(_pair_key(rec["source"], rec["target"]))
                done_books.add(rec["book_id"])
        print(f"  resuming — {len(already_seen)} pairs already classified")

    total = len(GUTENBERG_BOOKS)
    with out_path.open("a", encoding="utf-8") as out_file:
        for i, book in enumerate(GUTENBERG_BOOKS, 1):
            bid = str(book["id"])
            if bid in done_books:
                print(f"  [{i}/{total}] {book['title']} — already done, skipping")
                continue
            concepts = all_concepts.get(bid, [])
            text = (DATA_PROCESSED / f"{book['id']}.txt").read_text(encoding="utf-8")
            n = min(len(concepts), MAX_CONCEPTS_USED)
            print(f"  [{i}/{total}] {book['title']} ({n} concepts)")

            edges = extract_relations_for_book(book["id"], concepts, text, already_seen)

            for edge in edges:
                out_file.write(json.dumps(edge) + "\n")
                already_seen.add(_pair_key(edge["source"], edge["target"]))
            out_file.flush()

            rel_counts = {}
            for e in edges:
                rel_counts[e["relation"]] = rel_counts.get(e["relation"], 0) + 1
            print(f"    → {len(edges)} edges: {rel_counts}")

    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    type_all_relations()
