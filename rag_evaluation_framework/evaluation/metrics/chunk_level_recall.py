from rag_evaluation_framework.evaluation.metrics.base import Metrics
from typing import List, Optional
from langsmith.schemas import Example, Run

class ChunkLevelRecall(Metrics):
    def calculate(self, retrieved_chunk_ids: List[str], ground_truth_chunk_ids: List[str]) -> float:
        retrieved_chunk_ids_set = set(retrieved_chunk_ids)
        ground_truth_chunk_ids_set = set(ground_truth_chunk_ids)

        if len(ground_truth_chunk_ids_set) == 0 or len(retrieved_chunk_ids_set) == 0:
            return 0.0

        return len(retrieved_chunk_ids_set & ground_truth_chunk_ids_set) / len(ground_truth_chunk_ids_set)

    def extract_ground_truth_chunks_ids(self, example: Optional[Example]) -> List[str]:
        """Extract ground truth chunk IDs from Langsmith Example."""
        if example is None:
            return []
        
        # Try to get chunk_ids from outputs
        if hasattr(example, 'outputs') and example.outputs:
            if isinstance(example.outputs, dict):
                return example.outputs.get("chunk_ids", [])
            elif isinstance(example.outputs, list):
                return example.outputs
        
        return []

    def extract_retrieved_chunks_ids(self, run: Run) -> List[str]:
        """Extract retrieved chunk IDs from Langsmith Run."""
        if hasattr(run, 'outputs'):
            # The outputs should be a list of strings (chunks) from __run_retrieval
            if isinstance(run.outputs, list):
                return run.outputs
            elif isinstance(run.outputs, dict):
                # If outputs is a dict, try to get chunks or retrieved_chunks
                return run.outputs.get("chunks", run.outputs.get("retrieved_chunks", []))
        
        return []