from abc import ABC, abstractmethod
from typing import List

class Chunker(ABC):

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        raise NotImplementedError
