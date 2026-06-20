.PHONY: download clean-text fetch-metadata extract-concepts type-relations \
        build-graph inspect split-links cross-book-eval pipeline \
        train evaluate infer api frontend test lint

# ── data pipeline ─────────────────────────────────────────────────────────────

download:
	python -m src.ingestion.download_books

clean-text:
	python -m src.ingestion.clean_text

fetch-metadata:
	python -m src.ingestion.fetch_metadata

extract-concepts:
	python -m src.graph.extract_concepts

type-relations:
	python -m src.graph.type_relations

build-graph:
	python -m src.graph.build_graph

inspect:
	python -m src.graph.inspect_graph

split-links:
	python -m src.graph.split_links

cross-book-eval:
	python -m src.graph.build_cross_book_eval

# run the full graph-construction pipeline from scratch
pipeline: extract-concepts type-relations build-graph split-links cross-book-eval

# ── model ─────────────────────────────────────────────────────────────────────

train:
	python -m src.model.train

evaluate:
	python -m src.model.evaluate

infer:
	python -m src.model.inference

# ── serving ───────────────────────────────────────────────────────────────────

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run src/frontend/app.py --server.port 8501

# run API + frontend together (requires two terminals, or use & in shell)
serve: api

# ── dev ───────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
