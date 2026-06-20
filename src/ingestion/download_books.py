"""Download plain-text books from Project Gutenberg."""
import requests
from pathlib import Path
from src.config import DATA_RAW, GUTENBERG_BOOKS

GUTENBERG_URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
GUTENBERG_URL_ALT = "https://www.gutenberg.org/files/{id}/{id}.txt"


def download_book(book_id: int, title: str) -> Path:
    out_path = DATA_RAW / f"{book_id}.txt"
    if out_path.exists():
        print(f"  already downloaded: {title}")
        return out_path

    for url_template in [GUTENBERG_URL, GUTENBERG_URL_ALT]:
        url = url_template.format(id=book_id)
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            out_path.write_bytes(response.content)
            print(f"  downloaded: {title} ({out_path.stat().st_size // 1024} KB)")
            return out_path

    raise RuntimeError(f"Could not download book {book_id}: {title}")


def download_all() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for book in GUTENBERG_BOOKS:
        download_book(book["id"], book["title"])


if __name__ == "__main__":
    download_all()
