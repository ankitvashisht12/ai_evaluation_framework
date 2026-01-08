from abc import ABC, abstractmethod
from typing import List

class Embedder(ABC):

    @abstractmethod
    def embed_docs(self, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError
