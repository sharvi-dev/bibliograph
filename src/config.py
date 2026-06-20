import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_GRAPH = ROOT / "data" / "graph"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "models/gemini-2.5-flash"  # used for relation typing and /explain endpoint

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3.1:8b"

SPACY_MODEL = "en_core_web_lg"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# spaCy entity label categories to exclude from concept nodes
EXCLUDED_ENTITY_LABELS: set[str] = {
    "PERSON",    # people → belong in author nodes, not concept nodes
    "GPE",       # geopolitical entities (countries, cities)
    "LOC",       # locations
    "NORP",      # nationalities / religious / political groups
    "DATE",      # dates and periods
    "TIME",      # times
    "CARDINAL",  # numbers
    "ORDINAL",   # ordinal numbers
    "MONEY",     # monetary values
    "PERCENT",   # percentages
    "QUANTITY",  # measurements
    "FAC",       # facilities (buildings, airports)
    "LANGUAGE",  # languages
}

# single-word concepts that are too generic to be meaningful as standalone nodes
GENERIC_SINGLE_CONCEPTS: set[str] = {
    "thing", "case", "view", "manner", "part", "number",
    "way", "end", "kind", "sort", "use", "form", "order",
    "place", "said",
}

CONCEPT_BLOCKLIST: set[str] = {
    # cardinal and ordinal numbers
    "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "hundred", "thousand",
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "secondly", "thirdly", "fourthly",
    "the one",
    # generic person references
    "a man", "any man", "no man", "the people", "all men", "all mankind",
    # vague demonstrative phrases
    "all things", "those things", "these things", "the things", "any thing",
    "all ages",
    # generic placeholder phrases
    "the case", "this case", "this kind", "this manner", "this world",
    "this life", "this subject", "this view", "the view", "other hand",
    "greater part", "same manner", "present day", "the number",
    "every one", "no idea",
    # archaic pronouns
    "thou", "thy", "thee", "thine", "that thou",
    # publication and edition artifacts
    "chapter", "page", "section", "book", "author", "edition",
    "volume", "note", "footnote", "ibid", "b.c.", "b.c",
}

GUTENBERG_BOOKS = [
    {"id": 3300, "title": "The Wealth of Nations", "author": "Adam Smith"},
    {"id": 2680, "title": "Meditations", "author": "Marcus Aurelius"},
    {"id": 1497, "title": "The Republic", "author": "Plato"},
    {"id": 1228, "title": "On the Origin of Species", "author": "Charles Darwin"},
    {"id": 1232, "title": "The Prince", "author": "Niccolo Machiavelli"},
    {"id": 3207, "title": "Leviathan", "author": "Thomas Hobbes"},
    {"id": 9662, "title": "An Enquiry Concerning Human Understanding", "author": "David Hume"},
    {"id": 1998, "title": "Thus Spoke Zarathustra", "author": "Friedrich Nietzsche"},
    {"id": 132,  "title": "The Art of War", "author": "Sun Tzu"},
]

RELATION_TYPES = ["causes", "contradicts", "exemplifies", "extends", "is_a", "related_to"]

GRAPH_SAVE_PATH = DATA_GRAPH / "bibliograph.pt"
MODEL_SAVE_PATH = ROOT / "model_best.pt"
