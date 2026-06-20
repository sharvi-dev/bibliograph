---
title: BiblioGraph
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# BiblioGraph

A heterogeneous Graph Transformer that builds a cross-book knowledge graph from your reading history and discovers latent conceptual connections across books using link prediction.

---

## Repository structure

```
bibliograph/
├── data/
│   ├── raw/                        # downloaded .txt files from Gutenberg
│   ├── processed/                  # cleaned text, concepts, relations, metadata
│   └── graph/                      # saved PyG HeteroData objects
├── src/
│   ├── config.py                   # paths, model names, book list, blocklist
│   ├── ingestion/
│   │   ├── download_books.py       # pulls from Project Gutenberg
│   │   ├── fetch_metadata.py       # Open Library API → metadata.json
│   │   └── clean_text.py           # strip Gutenberg headers/footers
│   ├── graph/
│   │   ├── extract_concepts.py     # spaCy entity + noun phrase extraction
│   │   ├── type_relations.py       # Ollama LLM relation classification
│   │   ├── build_graph.py          # assembles PyG HeteroData object
│   │   └── inspect_graph.py        # NetworkX stats for debugging
│   ├── model/
│   │   ├── dataset.py              # link prediction dataset, negative sampling
│   │   ├── model.py                # TransformerConv architecture
│   │   ├── train.py                # training loop
│   │   └── evaluate.py             # AUC, precision@k, top bridge surfacing
│   ├── api/
│   │   ├── main.py                 # FastAPI app
│   │   ├── routes.py               # /bridges, /graph, /explain endpoints
│   │   └── schemas.py              # Pydantic request/response models
│   └── frontend/
│       └── app.py                  # Streamlit app
├── notebooks/
│   ├── 01_explore_graph.ipynb      # sanity checks on extracted graph
│   ├── 02_model_experiments.ipynb  # hyperparameter exploration
│   └── 03_bridge_analysis.ipynb    # qualitative analysis of top bridges
├── tests/
│   ├── test_extraction.py
│   ├── test_graph_build.py
│   └── test_model.py
├── .env.example
├── .gitignore
├── requirements.txt
├── Makefile
└── README.md
```
