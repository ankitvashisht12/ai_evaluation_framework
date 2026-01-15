from typing import List, Optional

from langsmith.schemas import Example, Run

from rag_evaluation_framework.evaluation.metrics.base import Metrics


class TokenLevelRecall(Metrics):
    """
    Token-level recall metric for RAG evaluation.
    
    Measures the proportion of tokens in the ground truth that appear
    in the retrieved chunks. This is useful when you want to measure
    how much of the relevant content was retrieved, regardless of
    chunk boundaries.
    """

    def __init__(self, case_sensitive: bool = False):
        """
        Initialize TokenLevelRecall metric.
        
        Args:
            case_sensitive: Whether token comparison should be case-sensitive.
                           Defaults to False (case-insensitive).
        """
        self.case_sensitive = case_sensitive

    def calculate(
        self, 
        retrieved_chunk_ids: List[str], 
        ground_truth_chunk_ids: List[str]
    ) -> float:
        """
        Calculate token-level recall.
        
        Note: This method receives chunk IDs but for token-level recall,
        we need the actual text content. In practice, you would need to
        pass the text content through the extract methods.
        
        Args:
            retrieved_chunk_ids: List of retrieved chunk texts/IDs
            ground_truth_chunk_ids: List of ground truth chunk texts/IDs
            
        Returns:
            Token-level recall score (0.0 to 1.0)
        """
        if not ground_truth_chunk_ids:
            return 0.0
        
        if not retrieved_chunk_ids:
            return 0.0
        
        # Tokenize ground truth
        ground_truth_text = " ".join(ground_truth_chunk_ids)
        ground_truth_tokens = self._tokenize(ground_truth_text)
        
        if not ground_truth_tokens:
            return 0.0
        
        # Tokenize retrieved content
        retrieved_text = " ".join(retrieved_chunk_ids)
        retrieved_tokens = self._tokenize(retrieved_text)
        
        # Calculate overlap
        ground_truth_set = set(ground_truth_tokens)
        retrieved_set = set(retrieved_tokens)
        
        intersection = ground_truth_set & retrieved_set
        
        return len(intersection) / len(ground_truth_set)

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenization.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not self.case_sensitive:
            text = text.lower()
        
        # Simple whitespace tokenization, removing empty strings
        tokens = [token.strip() for token in text.split()]
        return [t for t in tokens if t]

    def extract_ground_truth_chunks_ids(self, example: Optional[Example]) -> List[str]:
        """
        Extract ground truth content from Langsmith Example.
        
        Args:
            example: Langsmith Example containing ground truth data
            
        Returns:
            List of ground truth texts/chunk IDs
        """
        if example is None:
            return []
        
        # Try to get chunk content or chunk_ids from outputs
        if hasattr(example, 'outputs') and example.outputs:
            if isinstance(example.outputs, dict):
                # Prefer actual text content for token-level comparison
                if "chunks" in example.outputs:
                    return example.outputs["chunks"]
                if "chunk_text" in example.outputs:
                    return example.outputs["chunk_text"]
                return example.outputs.get("chunk_ids", [])
            elif isinstance(example.outputs, list):
                return example.outputs
        
        return []

    def extract_retrieved_chunks_ids(self, run: Run) -> List[str]:
        """
        Extract retrieved content from Langsmith Run.
        
        Args:
            run: Langsmith Run containing retrieval results
            
        Returns:
            List of retrieved texts/chunk IDs
        """
        if hasattr(run, 'outputs'):
            if isinstance(run.outputs, list):
                return run.outputs
            elif isinstance(run.outputs, dict):
                # Prefer actual text content
                if "chunks" in run.outputs:
                    return run.outputs["chunks"]
                if "retrieved_chunks" in run.outputs:
                    return run.outputs["retrieved_chunks"]
                return run.outputs.get("chunk_ids", [])
        
        return []
