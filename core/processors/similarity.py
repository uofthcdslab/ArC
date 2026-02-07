"""Similarity computation processor"""
from typing import List
from utils import helpers as hp


class SimilarityProcessor:
    """Handles semantic similarity computations"""
    
    def __init__(self, model_name: str, logger=None):
        """
        Initialize similarity processor
        
        Args:
            model_name: Name of the similarity model to use
            logger: Optional logger instance
        """
        self.logger = logger
        self.sims_hp = hp.SentenceSimilarity(model_name, logger)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        return float(self.sims_hp.predict([(text1, text2)])[0])
    
    def compute_between_runs_similarity(self, one_reason: str, 
                                       target_reasons: List[str],
                                       target_confidences: List[float],
                                       diversity: bool = True) -> float:
        """
        Compute similarity/diversity between one reason and target reasons
        
        Args:
            one_reason: The reason to compare
            target_reasons: List of target reasons
            target_confidences: Confidence scores for target reasons
            diversity: If True, compute diversity (1 - similarity)
            
        Returns:
            Weighted average similarity/diversity score
        """
        num = 0
        den = 0
        
        for target_reason, target_confidence in zip(target_reasons, target_confidences):
            sim = float(self.sims_hp.predict([(one_reason, target_reason)])[0])
            if diversity:
                sim = 1.0 - sim
            num += (sim * target_confidence)
            den += target_confidence
        
        return num / den if den > 0 else 0.0
