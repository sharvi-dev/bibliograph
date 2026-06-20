"""Streamlit frontend for Bibliograph."""
import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Bibliograph", layout="wide")
st.title("Bibliograph — Cross-Book Knowledge Explorer")

# TODO: populate from /graph endpoint or a local books list
BOOKS = [b["title"] for b in [
    {"title": "The Wealth of Nations"},
    {"title": "Meditations"},
    {"title": "The Republic"},
    {"title": "On the Origin of Species"},
    {"title": "The Prince"},
    {"title": "Leviathan"},
    {"title": "An Enquiry Concerning Human Understanding"},
    {"title": "Thus Spoke Zarathustra"},
    {"title": "The Art of War"},
]]

tab_bridges, tab_graph = st.tabs(["Hidden Bridges", "Graph Explorer"])

with tab_bridges:
    col1, col2 = st.columns(2)
    with col1:
        book_a = st.selectbox("Book A", BOOKS, key="book_a")
    with col2:
        book_b = st.selectbox("Book B", [b for b in BOOKS if b != book_a], key="book_b")
    top_k = st.slider("Top K bridges", 5, 50, 10)

    if st.button("Find hidden bridges"):
        resp = requests.get(f"{API_BASE}/bridges", params={"book_a": book_a, "book_b": book_b, "top_k": top_k})
        if resp.ok:
            for bridge in resp.json()["bridges"]:
                with st.container(border=True):
                    st.markdown(f"**{bridge['concept_a']}** ({bridge['book_a']}) ↔ **{bridge['concept_b']}** ({bridge['book_b']})")
                    st.caption(f"Confidence: {bridge['score']:.3f}")
                    if bridge.get("explanation"):
                        st.write(bridge["explanation"])
        else:
            st.error(f"API error: {resp.status_code}")

with tab_graph:
    selected_book = st.selectbox("Select a book to explore", BOOKS, key="graph_book")
    if st.button("Load graph"):
        st.info("Graph explorer coming soon — requires streamlit-agraph integration.")
