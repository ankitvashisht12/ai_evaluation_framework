from rag_evaluation_framework.evaluation.chunker.base import Chunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RecursiveCharTextSplitter(Chunker):
    chunk_size: int
    chunk_overlap: int

    def __init__(self, chunk_size: int=100, chunk_overlap: int=10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return text_splitter.split_text(text)
