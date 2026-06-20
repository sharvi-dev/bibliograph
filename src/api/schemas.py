from pydantic import BaseModel


class Bridge(BaseModel):
    concept_a: str
    concept_b: str
    book_a: str
    book_b: str
    score: float
    explanation: str | None = None


class BridgesResponse(BaseModel):
    bridges: list[Bridge]


class GraphNode(BaseModel):
    id: str
    label: str
    node_type: str
    book: str | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class ExplainRequest(BaseModel):
    concept_a: str
    book_a: str
    concept_b: str
    book_b: str


class ExplainResponse(BaseModel):
    explanation: str
