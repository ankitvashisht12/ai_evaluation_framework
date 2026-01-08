
from rag_evaluation_framework.evaluation.vector_store.base import VectorStore
from typing import List

class ChromaVectorStore(VectorStore):

    def __init__(self):
        pass

    def embed_docs(self, docs: List[str]) -> List[List[float]]:
        return []

    def search(self, query: str, k: int) -> List[str]:
        return []
