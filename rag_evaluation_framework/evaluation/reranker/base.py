from abc import ABC, abstractmethod
from typing import List


class Reranker(ABC):

    @abstractmethod
    def rerank(self, docs: List[str], query: str, k: int) -> List[str]:
        raise NotImplementedError