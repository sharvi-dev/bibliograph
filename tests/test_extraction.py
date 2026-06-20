"""Tests for concept extraction pipeline."""
import pytest
from src.ingestion.clean_text import clean_book
from pathlib import Path


def test_clean_strips_gutenberg_header(tmp_path):
    raw = tmp_path / "test.txt"
    raw.write_text(
        "Some preamble\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK FOO ***\n\n"
        "Actual content here.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK FOO ***\n\n"
        "Some footer"
    )
    cleaned = clean_book(raw)
    assert "Actual content here." in cleaned
    assert "preamble" not in cleaned
    assert "footer" not in cleaned


def test_clean_collapses_blank_lines(tmp_path):
    raw = tmp_path / "test.txt"
    raw.write_text("Line one.\n\n\n\n\nLine two.")
    cleaned = clean_book(raw)
    assert "\n\n\n" not in cleaned
