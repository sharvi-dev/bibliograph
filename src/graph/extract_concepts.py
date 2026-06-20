"""Extract concept nodes from cleaned book text using spaCy.

Pipeline order (per the document):
  Extract spans → exclude entity labels → normalize → apply blocklist
  → count canonical concepts → enforce min frequency → select top N

The 150-concept cap is applied LAST so that filtering noise allows
valid lower-ranked concepts to move up into the retained set.
"""
import json
import re
import spacy
from src.config import (
    DATA_PROCESSED,
    SPACY_MODEL,
    CONCEPT_BLOCKLIST,
    EXCLUDED_ENTITY_LABELS,
    GENERIC_SINGLE_CONCEPTS,
    GUTENBERG_BOOKS,
)

MIN_CONCEPT_FREQ = 3
MAX_CONCEPTS_PER_BOOK = 150

_LEADING_ARTICLES = ("the ", "a ", "an ")
_LEADING_POSSESSIVES = ("my ", "your ", "his ", "her ", "its ", "their ", "our ", "thy ")
# generic determiners/quantifiers that make noun chunks too vague when leading
_LEADING_DETERMINERS = (
    "any ", "that ", "every ", "some ", "other ", "no ",
    "all ", "such ", "this ", "each ", "both ", "same ",
)
_TITLE_PREFIXES = ("dr.", "mr.", "mrs.", "prof.", "sir ", "lord ")
_ROMAN = re.compile(r"^[ivxlcdmIVXLCDM]+$")
_PUNCT_ONLY = re.compile(r"^[\W_]+$")


def _normalize(text: str) -> str:
    """Full normalization pipeline: lowercase → collapse whitespace →
    strip leading articles/possessives → strip trailing possessives →
    remove bracket artifacts, purely numeric, and Roman numerals.
    """
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)           # collapse internal whitespace / newlines

    for art in _LEADING_ARTICLES:
        if t.startswith(art):
            t = t[len(art):]
            break

    for poss in _LEADING_POSSESSIVES:
        if t.startswith(poss):
            t = t[len(poss):]
            break

    for det in _LEADING_DETERMINERS:
        if t.startswith(det):
            t = t[len(det):]
            break

    t = re.sub(r"'s$", "", t)            # strip trailing possessives ("sun tzŭ's" → "sun tzŭ")
    t = t.strip(".,;:\"'()[]{}")          # strip surrounding punctuation

    if not t:
        return ""
    if t.startswith("[") or t.startswith("]"):
        return ""
    if _PUNCT_ONLY.match(t):
        return ""
    if t.isdigit():
        return ""
    if _ROMAN.match(t):                   # filter Roman numeral chapter headings
        return ""

    return t


def should_reject_concept(text: str, ent_label: str | None = None) -> bool:
    """Return True if the span should be discarded before counting."""
    if not text or len(text) <= 2:
        return True
    if ent_label and ent_label in EXCLUDED_ENTITY_LABELS:
        return True
    if text in CONCEPT_BLOCKLIST:
        return True
    words = text.split()
    if len(words) == 1 and text in GENERIC_SINGLE_CONCEPTS:
        return True
    # reject honorific-prefixed names that slipped past NER (e.g. "dr. hooker")
    if any(text.startswith(p) for p in _TITLE_PREFIXES):
        return True
    # reject "chapter i/ii/iii/..." and "section i/ii/..." structural artifacts
    first_word = text.split()[0] if text.split() else ""
    if first_word in ("chapter", "section", "book", "part", "volume", "appendix"):
        return True
    return False


def load_nlp():
    return spacy.load(SPACY_MODEL)


def get_paragraphs(text: str, max_chars: int = 5_000) -> list[str]:
    return [p.strip()[:max_chars] for p in text.split("\n\n") if p.strip()]


def extract_concepts_from_book(book_id: int, nlp) -> list[str]:
    text = (DATA_PROCESSED / f"{book_id}.txt").read_text(encoding="utf-8")
    paragraphs = get_paragraphs(text)

    freq: dict[str, int] = {}

    for doc in nlp.pipe(paragraphs, batch_size=64):
        # build a set of character spans covered by excluded entity types
        excluded_spans: set[tuple[int, int]] = set()
        for ent in doc.ents:
            if ent.label_ in EXCLUDED_ENTITY_LABELS:
                excluded_spans.add((ent.start_char, ent.end_char))

        # named entities that pass label filter
        for ent in doc.ents:
            if ent.label_ in EXCLUDED_ENTITY_LABELS:
                continue
            token = _normalize(ent.text)
            if should_reject_concept(token, ent.label_):
                continue
            freq[token] = freq.get(token, 0) + 1

        # noun chunks that don't overlap with excluded entity spans
        for nc in doc.noun_chunks:
            # skip if the noun chunk is fully covered by an excluded entity
            if any(
                nc.start_char >= s and nc.end_char <= e
                for s, e in excluded_spans
            ):
                continue
            token = _normalize(nc.text)
            words = token.split()
            if len(words) < 2 or len(words) > 4:
                continue
            if should_reject_concept(token):
                continue
            freq[token] = freq.get(token, 0) + 1

    # cap is applied LAST — after all filtering
    ranked = sorted(
        ((c, f) for c, f in freq.items() if f >= MIN_CONCEPT_FREQ),
        key=lambda x: x[1],
        reverse=True,
    )
    return [c for c, _ in ranked[:MAX_CONCEPTS_PER_BOOK]]


def extract_all(nlp=None) -> None:
    if nlp is None:
        nlp = load_nlp()
    out: dict[str, list[str]] = {}
    total = len(GUTENBERG_BOOKS)
    for i, book in enumerate(GUTENBERG_BOOKS, 1):
        print(f"  [{i}/{total}] {book['title']}")
        concepts = extract_concepts_from_book(book["id"], nlp)
        out[str(book["id"])] = concepts
        print(f"    → {len(concepts)} concepts")

    out_path = DATA_PROCESSED / "concepts.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    extract_all()
