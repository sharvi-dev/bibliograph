"""Strip Project Gutenberg headers/footers, normalize whitespace, and emit paragraphs.jsonl."""
import json
import re
import unicodedata
from pathlib import Path
from src.config import DATA_RAW, DATA_PROCESSED, GUTENBERG_BOOKS

START_PATTERN = re.compile(
    r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE
)
END_PATTERN = re.compile(
    r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE
)
# running headers / page numbers that repeat on their own line
PAGE_LINE = re.compile(r"^\s*\d+\s*$", re.MULTILINE)


BRACKET_COMMENTARY = re.compile(r"\[[^\]]{1,500}\]")  # strip [bracketed commentary]
# attribution lines: short lines that are just a commentator name followed by a colon
# e.g. "Ts'ao Kung:", "_Tu Mu_:", "Wang Hsi:" in Art of War
ATTRIBUTION_LINE = re.compile(r"^_?[\w\s''éàü\-\.]{1,40}_?:\s*$", re.MULTILINE)


def _strip_gutenberg(text: str) -> str:
    start = START_PATTERN.search(text)
    end = END_PATTERN.search(text)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]
    return text


def _strip_brackets(text: str) -> str:
    """Remove [bracketed commentary/footnotes] and inline attribution lines."""
    text = BRACKET_COMMENTARY.sub("", text)
    text = ATTRIBUTION_LINE.sub("", text)
    return text


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _repair_line_wraps(text: str) -> str:
    """Re-join lines that were wrapped mid-sentence (no blank line between them)."""
    # a blank line = paragraph break; single newline = possible wrap
    # join wrapped lines with a space, preserve paragraph breaks
    lines = text.split("\n")
    out, buf = [], []
    for line in lines:
        stripped = line.strip()
        if stripped:
            buf.append(stripped)
        else:
            if buf:
                out.append(" ".join(buf))
                buf = []
            out.append("")
    if buf:
        out.append(" ".join(buf))
    return "\n".join(out)


def clean_book(raw_path: Path) -> str:
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    text = _strip_gutenberg(text)
    text = _normalize_unicode(text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = PAGE_LINE.sub("", text)
    text = _strip_brackets(text)
    text = _repair_line_wraps(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _book_id_slug(book: dict) -> str:
    return book["title"].lower().replace(" ", "_").replace(",", "").replace("'", "")


def paragraphs_from_text(text: str, book_slug: str) -> list[dict]:
    """Split cleaned text into paragraphs with deterministic IDs."""
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    records = []
    for idx, para in enumerate(raw_paras):
        # skip very short paragraphs (likely headings or artifacts, < 40 chars)
        if len(para) < 40:
            continue
        records.append({
            "paragraph_id": f"{book_slug}_{idx:06d}",
            "book_slug": book_slug,
            "paragraph_index": idx,
            "text": para,
        })
    return records


def clean_all() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_PROCESSED / "paragraphs.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for book in GUTENBERG_BOOKS:
            raw_path = DATA_RAW / f"{book['id']}.txt"
            if not raw_path.exists():
                print(f"  missing raw file: {book['title']}, skipping")
                continue

            cleaned = clean_book(raw_path)
            slug = _book_id_slug(book)
            paras = paragraphs_from_text(cleaned, slug)

            # write cleaned .txt for human inspection
            txt_out = DATA_PROCESSED / f"{book['id']}.txt"
            txt_out.write_text(cleaned, encoding="utf-8")

            # write JSONL records
            for rec in paras:
                jsonl_file.write(json.dumps(rec) + "\n")

            lengths = [len(p["text"]) for p in paras]
            median_len = sorted(lengths)[len(lengths) // 2] if lengths else 0
            print(
                f"  {book['title']}: {len(paras)} paragraphs, "
                f"median {median_len} chars, max {max(lengths) if lengths else 0} chars"
            )

    print(f"\n  saved → {jsonl_path}")


if __name__ == "__main__":
    clean_all()
