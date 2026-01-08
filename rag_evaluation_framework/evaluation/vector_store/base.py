from typing import List
from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    def embed_docs(self, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, k: int) -> List[str]:
        raise NotImplementedError
