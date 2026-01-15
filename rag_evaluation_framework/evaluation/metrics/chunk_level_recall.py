import logging
from rag_evaluation_framework.evaluation.metrics.base import Metrics
from typing import List, Optional
from langsmith.schemas import Example, Run

logger = logging.getLogger(__name__)

class ChunkLevelRecall(Metrics):
    def calculate(self, retrieved_chunk_ids: List[str], ground_truth_chunk_ids: List[str]) -> float:
        logger.debug(
            "Calculating chunk-level recall with retrieved=%d, ground_truth=%d",
            len(retrieved_chunk_ids),
            len(ground_truth_chunk_ids),
        )
        logger.debug("Retrieved chunk IDs: %s", retrieved_chunk_ids)
        logger.debug("Ground truth chunk IDs: %s", ground_truth_chunk_ids)
        retrieved_chunk_ids_set = set(retrieved_chunk_ids)
        ground_truth_chunk_ids_set = set(ground_truth_chunk_ids)

        if len(ground_truth_chunk_ids_set) == 0 or len(retrieved_chunk_ids_set) == 0:
            logger.debug("Empty retrieved or ground truth set; returning 0.0")
            return 0.0

        score = len(retrieved_chunk_ids_set & ground_truth_chunk_ids_set) / len(ground_truth_chunk_ids_set)
        logger.debug("Chunk-level recall score: %s", score)
        return score

    def extract_ground_truth_chunks_ids(self, example: Optional[Example]) -> List[str]:
        """Extract ground truth chunk IDs from Langsmith Example."""
        if example is None:
            logger.debug("No example provided; ground truth chunk IDs empty")
            return []
        
        logger.debug("Extracting ground truth chunk IDs from example: %s", type(example).__name__)
        if hasattr(example, "outputs"):
            logger.debug("Example outputs type: %s", type(example.outputs).__name__)
            logger.debug("Example outputs: %s", example.outputs)
        
        # Try to get chunk_ids from outputs
        if hasattr(example, 'outputs') and example.outputs:
            if isinstance(example.outputs, dict):
                chunk_ids = example.outputs.get("chunk_ids", [])
                logger.debug("Ground truth chunk IDs from outputs['chunk_ids']: %s", chunk_ids)
                return chunk_ids
            elif isinstance(example.outputs, list):
                logger.debug("Ground truth chunk IDs from outputs list: %s", example.outputs)
                return example.outputs
        
        logger.debug("No ground truth chunk IDs found in example outputs")
        return []

    def extract_retrieved_chunks_ids(self, run: Run) -> List[str]:
        """Extract retrieved chunk IDs from Langsmith Run."""
        if hasattr(run, 'outputs'):
            logger.debug("Extracting retrieved chunk IDs from run outputs type: %s", type(run.outputs).__name__)
            logger.debug("Run outputs: %s", run.outputs)
            # The outputs should be a list of strings (chunks) from __run_retrieval
            if isinstance(run.outputs, list):
                logger.debug("Retrieved chunk IDs from outputs list: %s", run.outputs)
                return run.outputs
            elif isinstance(run.outputs, dict):
                # If outputs is a dict, try to get chunks or retrieved_chunks
                chunk_ids = run.outputs.get("chunks", run.outputs.get("retrieved_chunks", []))
                logger.debug("Retrieved chunk IDs from outputs dict: %s", chunk_ids)
                return chunk_ids
        
        logger.debug("No retrieved chunk IDs found in run outputs")
        return []