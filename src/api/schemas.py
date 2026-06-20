from pydantic import BaseModel


class Bridge(BaseModel):
    concept_a: str
    concept_b: str
    books_a: list[str]
    books_b: list[str]
    score: float
    evidence: str = ""
    relation: str = ""


class BridgesResponse(BaseModel):
    bridges: list[Bridge]
    book_a_slug: str
    book_b_slug: str


class GraphNode(BaseModel):
    id: str
    label: str
    node_type: str   # "concept" | "book" | "author"
    books: list[str] = []


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class ExplainRequest(BaseModel):
    concept_a: str
    books_a: list[str]
    concept_b: str
    books_b: list[str]
    score: float = 0.0
    evidence: str = ""


class ExplainResponse(BaseModel):
    explanation: str
