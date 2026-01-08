from pathlib import Path
from typing import Callable, Optional
from rag_evaluation_framework.evaluation.chunker.base import Chunker
from rag_evaluation_framework.evaluation.vector_store.base import VectorStore
from rag_evaluation_framework.evaluation.reranker.base import Reranker
from rag_evaluation_framework.evaluation.embedder.base import Embedder

class Evaluation:

    langsmith_dataset_name: str
    kb_data_path: str

    def __init__(self, langsmith_dataset_name: str, kb_data_path: str):
        self.langsmith_dataset_name = langsmith_dataset_name
        self.kb_data_path = kb_data_path


    def run(
        self,
        chunker: Optional[Chunker] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        k: int = 5,
        reranker: Optional[Reranker] = None,
    ):
        if not self.langsmith_dataset_name:
            raise ValueError("langsmith_dataset_name is required")

        if not self.kb_data_path:
            raise ValueError("kb_data_path is required")

        
