from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

class Chunk(BaseModel):
    text: str
    start_index: int
    end_index: int

class Chunker(ABC):

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        raise NotImplementedError
