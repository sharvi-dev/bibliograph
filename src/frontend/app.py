"""BiblioGraph — Streamlit frontend."""
import requests
import streamlit as st

import os
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="BiblioGraph",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.bridge-card { border:1px solid #e0e0e0; border-radius:8px; padding:16px; margin-bottom:12px; }
.score-badge { background:#1a73e8; color:white; border-radius:12px;
               padding:2px 10px; font-size:0.85em; font-weight:600; }
.concept-tag { background:#f1f3f4; border-radius:4px;
               padding:2px 8px; font-family:monospace; font-size:0.9em; }
.book-tag { color:#5f6368; font-size:0.8em; }
</style>
""", unsafe_allow_html=True)

st.title("📚 BiblioGraph")
st.caption("Cross-book knowledge discovery via heterogeneous Graph Transformer")

# ── fetch book list ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_books():
    try:
        r = requests.get(f"{API_BASE}/books", timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    # fallback if API not yet running
    titles = [
        "The Wealth of Nations", "Meditations", "The Republic",
        "On the Origin of Species", "The Prince", "Leviathan",
        "An Enquiry Concerning Human Understanding",
        "Thus Spoke Zarathustra", "The Art of War",
    ]
    def _slug(t):
        return t.lower().replace(" ", "_").replace(",", "").replace("'", "")
    return [{"title": t, "slug": _slug(t)} for t in titles]

BOOKS = fetch_books()
TITLE_TO_SLUG = {b["title"]: b["slug"] for b in BOOKS}
TITLES = [b["title"] for b in BOOKS]

RELATION_COLORS = {
    "causes":      "#e74c3c",
    "contradicts": "#8e44ad",
    "exemplifies": "#27ae60",
    "extends":     "#2980b9",
    "is_a":        "#f39c12",
    "related_to":  "#95a5a6",
}

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_bridges, tab_graph, tab_about = st.tabs(["🔗 Hidden Bridges", "🕸️ Graph Explorer", "ℹ️ About"])


# ── BRIDGES TAB ───────────────────────────────────────────────────────────────
with tab_bridges:
    st.subheader("Find hidden conceptual bridges between two books")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        title_a = st.selectbox("Book A", TITLES, index=2, key="ba")  # default: The Republic
    with col2:
        remaining = [t for t in TITLES if t != title_a]
        title_b = st.selectbox("Book B", remaining, index=2, key="bb")  # default: On the Origin of Species
    with col3:
        top_k = st.slider("Top K", 5, 30, 10)

    exclusive = st.toggle(
        "Exclusive concepts only",
        value=True,
        help="When on, only surfaces concepts unique to each book — avoids hub concepts "
             "(like 'state' or 'one man') that span many books dominating the results.",
    )

    if st.button("🔍 Find bridges", type="primary", use_container_width=True):
        slug_a = TITLE_TO_SLUG[title_a]
        slug_b = TITLE_TO_SLUG[title_b]
        with st.spinner("Scoring 500K+ cross-book concept pairs…"):
            try:
                resp = requests.get(
                    f"{API_BASE}/bridges",
                    params={"book_a": slug_a, "book_b": slug_b, "top_k": top_k, "exclusive": str(exclusive).lower()},
                    timeout=30,
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Make sure it's running: `uvicorn src.api.main:app --reload`")
                st.stop()

        if not resp.ok:
            st.error(f"API error {resp.status_code}: {resp.text}")
            st.stop()

        data = resp.json()
        bridges = data["bridges"]

        if not bridges:
            st.info(f"No cross-book bridges found between **{title_a}** and **{title_b}**.")
        else:
            st.success(f"Found **{len(bridges)}** bridges between *{title_a}* and *{title_b}*")
            for i, b in enumerate(bridges, 1):
                with st.expander(
                    f"#{i}  **{b['concept_a']}** ↔ **{b['concept_b']}**   "
                    f"(confidence {b['score']:.3f})",
                    expanded=(i <= 3),
                ):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**{b['concept_a']}**")
                        st.caption(" · ".join(b["books_a"]))
                    with c2:
                        st.markdown(f"**{b['concept_b']}**")
                        st.caption(" · ".join(b["books_b"]))

                    if b.get("relation"):
                        st.markdown(
                            f"Relation detected: `{b['relation']}`",
                        )

                    if b.get("evidence"):
                        st.markdown("**Context snippet:**")
                        st.markdown(f"> {b['evidence'][:300]}")

                    if st.button("✨ Explain this bridge", key=f"explain_{i}"):
                        ex_resp = requests.post(
                            f"{API_BASE}/explain",
                            json={
                                "concept_a": b["concept_a"],
                                "books_a":   b["books_a"],
                                "concept_b": b["concept_b"],
                                "books_b":   b["books_b"],
                                "score":     b["score"],
                                "evidence":  b.get("evidence", ""),
                            },
                            timeout=15,
                        )
                        if ex_resp.ok:
                            st.info(ex_resp.json()["explanation"])
                        else:
                            st.warning("Explanation unavailable.")


# ── GRAPH TAB ─────────────────────────────────────────────────────────────────
with tab_graph:
    st.subheader("Explore the concept graph for a single book")

    selected_title = st.selectbox("Select book", TITLES, key="graph_book")
    slug = TITLE_TO_SLUG[selected_title]

    if st.button("📊 Load graph", type="primary"):
        with st.spinner("Loading concept graph…"):
            try:
                resp = requests.get(f"{API_BASE}/graph", params={"book": slug}, timeout=15)
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API.")
                st.stop()

        if not resp.ok:
            st.error(f"API error: {resp.status_code}")
            st.stop()

        gdata = resp.json()
        nodes = gdata["nodes"]
        edges = gdata["edges"]

        concept_nodes = [n for n in nodes if n["node_type"] == "concept"]
        book_nodes    = [n for n in nodes if n["node_type"] == "book"]

        st.markdown(
            f"**{len(concept_nodes)}** concepts · **{len(edges)}** semantic edges"
        )

        # try streamlit-agraph; graceful fallback to table
        try:
            from streamlit_agraph import agraph, Node, Edge, Config

            ag_nodes = []
            for n in nodes:
                if n["node_type"] == "book":
                    ag_nodes.append(Node(id=n["id"], label=n["label"],
                                         size=30, color="#f39c12", font={"size": 14, "bold": True}))
                else:
                    ag_nodes.append(Node(id=n["id"], label=n["label"],
                                         size=12, color="#1a73e8"))

            ag_edges = []
            for e in edges:
                color = RELATION_COLORS.get(e["relation"], "#bdc3c7")
                ag_edges.append(Edge(source=e["source"], target=e["target"],
                                     color=color, label=e["relation"]))

            config = Config(
                width="100%", height=600,
                directed=False, physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#f1c40f",
            )
            agraph(nodes=ag_nodes, edges=ag_edges, config=config)

        except ImportError:
            st.warning(
                "Install `streamlit-agraph` for the interactive graph view. "
                "Showing table instead."
            )
            import pandas as pd
            st.write("**Concepts:**")
            st.dataframe(
                pd.DataFrame([{"concept": n["label"]} for n in concept_nodes]),
                use_container_width=True, hide_index=True,
            )
            st.write("**Edges (sample of 50):**")
            # resolve ids to names
            id_to_label = {n["id"]: n["label"] for n in nodes}
            edge_rows = [
                {"source": id_to_label.get(e["source"], e["source"]),
                 "relation": e["relation"],
                 "target": id_to_label.get(e["target"], e["target"])}
                for e in edges[:50]
            ]
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)

        # relation breakdown
        if edges:
            from collections import Counter
            rel_counts = Counter(e["relation"] for e in edges)
            st.markdown("**Relation breakdown:**")
            cols = st.columns(len(rel_counts))
            for col, (rel, cnt) in zip(cols, sorted(rel_counts.items(), key=lambda x: -x[1])):
                col.metric(rel, cnt)


# ── ABOUT TAB ─────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
## What is BiblioGraph?

BiblioGraph is a **cross-book knowledge discovery engine** that builds a heterogeneous
knowledge graph from 9 classic texts and uses a **Graph Transformer** to find latent
conceptual connections — ideas that resonate across books that were never read together.

### How it works

1. **Ingestion** — 9 books from Project Gutenberg, cleaned and paragraph-segmented
2. **Concept extraction** — spaCy `en_core_web_lg` extracts named entities and noun phrases; top 100–150 per book after filtering
3. **Relation typing** — co-occurrence + keyword heuristics classify edges into 6 types: `causes`, `contradicts`, `exemplifies`, `extends`, `is_a`, `related_to`
4. **Graph construction** — PyTorch Geometric `HeteroData` with 3 node types (concept, book, author) and 10 edge types
5. **Graph Transformer** — `HeteroBibliographGT` with per-type input projections, `TransformerConv(concat=False)`, LayerNorm, and residual connections
6. **Link prediction** — dot-product (cosine similarity) decoder trained to distinguish real edges from random non-edges
7. **Bridge discovery** — after training, the model scores all 551K+ cross-book concept pairs; the highest-confidence pairs are the **hidden bridges**

### Model performance

| Metric | Value |
|---|---|
| Val AUC | 0.983 |
| Test AUC | 0.967 |
| Parameters | ~3M |
| Cross-book pairs scored | 551,366 |

### Books in the corpus

| Book | Author |
|---|---|
| The Wealth of Nations | Adam Smith |
| Meditations | Marcus Aurelius |
| The Republic | Plato |
| On the Origin of Species | Charles Darwin |
| The Prince | Niccolò Machiavelli |
| Leviathan | Thomas Hobbes |
| An Enquiry Concerning Human Understanding | David Hume |
| Thus Spoke Zarathustra | Friedrich Nietzsche |
| The Art of War | Sun Tzu |
""")
