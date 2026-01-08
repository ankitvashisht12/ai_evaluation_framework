from pathlib import Path
from typing import Callable, Optional
from rag_evaluation_framework.evaluation.chunker.base import Chunker
from rag_evaluation_framework.evaluation.vector_store.base import VectorStore
from rag_evaluation_framework.evaluation.reranker.base import Reranker
from rag_evaluation_framework.evaluation.embedder.base import Embedder

class Evaluation:
    def __init__(self):
        pass

    def run(
        self,
        langsmith_dataset_name: str,
        kb_data_path: Path,
        chunker: Optional[Chunker] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        k: int = 5,
        reranker: Optional[Reranker] = None,
    ):
       pass 