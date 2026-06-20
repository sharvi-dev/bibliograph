"""Fetch book metadata from Open Library API and save to data/processed/metadata.json."""
import json
import requests
from src.config import DATA_PROCESSED, GUTENBERG_BOOKS

OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"


def fetch_metadata(title: str, author: str) -> dict:
    params = {"title": title, "author": author, "limit": 1}
    response = requests.get(OPEN_LIBRARY_URL, params=params, timeout=15)
    response.raise_for_status()
    docs = response.json().get("docs", [])
    if not docs:
        return {}
    doc = docs[0]
    return {
        "title": doc.get("title", title),
        "author": doc.get("author_name", [author])[0],
        "year": doc.get("first_publish_year"),
        "subject": doc.get("subject", [])[:5],
        "open_library_key": doc.get("key"),
    }


def fetch_all() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for book in GUTENBERG_BOOKS:
        print(f"  fetching metadata: {book['title']}")
        meta = fetch_metadata(book["title"], book["author"])
        meta["gutenberg_id"] = book["id"]
        metadata[str(book["id"])] = meta

    out_path = DATA_PROCESSED / "metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  saved metadata for {len(metadata)} books → {out_path}")


if __name__ == "__main__":
    fetch_all()
