"""Classify relations between co-occurring concept pairs using Gemini.

Writes data/processed/relations.jsonl incrementally — one record per unique pair.
Resumes safely from the last completed pair on re-run.
Deduplicates: same (A, B) seen in multiple paragraphs → one record with best confidence.

Free-tier limits (gemini-2.0-flash): 15 RPM, ~1M tokens/day.
Rate limit: 12 RPM (5s between calls) for safety margin.
MAX_EDGES_PER_BOOK = 300 keeps total calls ~2700, ~950K tokens/day.
"""
import hashlib
import json
import random
import time
from google import genai
from google.genai import types
from src.config import (
    DATA_PROCESSED,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    RELATION_TYPES,
    GUTENBERG_BOOKS,
)

MAX_CONCEPTS_USED = 100
MAX_CONCEPTS_PER_PARA = 6
MAX_EDGES_PER_BOOK = 150   # 9 books * 150 = ~1350 calls at 5 RPM ≈ 4.5 hours
RATE_LIMIT_DELAY = 13.0    # seconds between calls → stays under 5 RPM free tier
MAX_RETRIES = 3

ALL_LABELS = RELATION_TYPES + ["none"]

RELATION_DEFINITIONS = "\n".join([
    "causes       — A produces or contributes to B",
    "contradicts  — A is incompatible with B (symmetric)",
    "exemplifies  — A is an example/manifestation of B (narrower → broader)",
    "extends      — A develops or expands B",
    "is_a         — A is a subtype or instance of B",
    "related_to   — meaningful connection without a more precise label (symmetric)",
    "none         — the paragraph does not clearly support any relation",
])

SYSTEM_PROMPT = f"""You classify semantic relations between philosophical and intellectual concepts.

Relation definitions:
{RELATION_DEFINITIONS}

Rules:
- Respond with valid JSON only.
- Set confidence between 0.0 and 1.0.
- evidence: one short sentence quoting or paraphrasing the supporting context.
- If unsure, use "related_to" rather than "none"."""

USER_PROMPT = """Concept A: "{concept_a}"
Concept B: "{concept_b}"
Context: "{context}"

Classify the relation from Concept A to Concept B."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "relation":   {"type": "STRING", "enum": ALL_LABELS},
        "confidence": {"type": "NUMBER"},
        "evidence":   {"type": "STRING"},
    },
    "required": ["relation", "confidence", "evidence"],
}


def _build_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def _pair_key(a: str, b: str) -> str:
    canonical = tuple(sorted([a, b]))
    return hashlib.md5("|".join(canonical).encode()).hexdigest()


def classify_relation(
    client: genai.Client,
    concept_a: str,
    concept_b: str,
    context: str,
) -> dict | None:
    prompt = USER_PROMPT.format(
        concept_a=concept_a,
        concept_b=concept_b,
        context=context[:250],
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.0,
                ),
            )
            time.sleep(RATE_LIMIT_DELAY)
            result = json.loads(response.text)
            label = result.get("relation", "none").lower().strip()
            if label not in ALL_LABELS:
                label = "related_to"
            return {
                "relation":   label,
                "confidence": float(result.get("confidence", 0.5)),
                "evidence":   result.get("evidence", ""),
            }
        except Exception as exc:
            msg = str(exc)
            # honour the retry delay the API suggests when present
            import re as _re
            m = _re.search(r"retry in (\d+)", msg)
            wait = int(m.group(1)) + 5 if m else RATE_LIMIT_DELAY * (2 ** attempt)
            print(f"    [warn] attempt {attempt} failed, retrying in {wait:.0f}s")
            time.sleep(wait)
    return None


def extract_relations_for_book(
    client: genai.Client,
    book_id: int,
    concepts: list[str],
    text: str,
    already_seen: set[str],
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
                result = classify_relation(client, a, b, para)
                if result is None or result["relation"] == "none":
                    continue
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

    client = _build_client()
    total = len(GUTENBERG_BOOKS)

    with out_path.open("a", encoding="utf-8") as out_file:
        for i, book in enumerate(GUTENBERG_BOOKS, 1):
            bid = str(book["id"])
            if bid in done_books:
                print(f"  [{i}/{total}] {book['title']} — already done, skipping")
                continue

            concepts = all_concepts.get(bid, [])
            text = (DATA_PROCESSED / f"{book['id']}.txt").read_text(encoding="utf-8")
            n_concepts = min(len(concepts), MAX_CONCEPTS_USED)
            print(f"  [{i}/{total}] {book['title']} ({n_concepts} concepts, max {MAX_EDGES_PER_BOOK} edges)")

            edges = extract_relations_for_book(client, book["id"], concepts, text, already_seen)

            for edge in edges:
                out_file.write(json.dumps(edge) + "\n")
                already_seen.add(_pair_key(edge["source"], edge["target"]))
            out_file.flush()
            print(f"    → {len(edges)} edges written")

    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    type_all_relations()
