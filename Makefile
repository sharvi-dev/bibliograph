.PHONY: download clean-text fetch-metadata extract-concepts type-relations build-graph inspect train evaluate api frontend test lint

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

train:
	python -m src.model.train

evaluate:
	python -m src.model.evaluate

api:
	uvicorn src.api.main:app --reload

frontend:
	streamlit run src/frontend/app.py

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
