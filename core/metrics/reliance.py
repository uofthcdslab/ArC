"""Reliance dimension metrics: UII and UEI"""
import torch
from typing import Dict, Any
from .base import BaseMetric
from utils import arc_hyperparams as arc_hp


class UIIMetric(BaseMetric):
    """Uncertainty in Internal Informativeness metric"""
    
    def __init__(self, similarity_processor, logger=None):
        super().__init__(logger)
        self.similarity_processor = similarity_processor
    
    def get_name(self) -> str:
        return "UII"
    
    def compute(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute UII for each internal reason
        
        Args:
            sample_data: Contains internal_reasons, internal_reasons_confidences,
                        initial_reasons, and initial_reasons_confidences
            
        Returns:
            Dictionary with UII scores
        """
        internal_reasons = sample_data.get('internal_reasons', [])
        internal_confidences = sample_data.get('internal_reasons_confidences', [])
        initial_reasons = sample_data.get('initial_reasons', [])
        initial_confidences = sample_data.get('initial_reasons_confidences', [])
        
        if not internal_reasons:
            self.log_warning("No internal reasons found for UII computation")
            return {}
        
        uii_scores = {}
        for reason_ix in range(len(internal_reasons)):
            confidence = internal_confidences[reason_ix]
            
            # Compute between-runs diversity
            between_runs_diversity = self.similarity_processor.compute_between_runs_similarity(
                internal_reasons[reason_ix],
                initial_reasons,
                initial_confidences,
                diversity=True
            )
            
            uii_score = (arc_hp.UII_Prediction_Weight * confidence) + \
                       (arc_hp.UII_Diversity_Weight * between_runs_diversity)
            
            uii_scores[f'reason_{reason_ix}'] = uii_score
        
        return uii_scores


class UEIMetric(BaseMetric):
    """Uncertainty in External Informativeness metric"""
    
    def __init__(self, similarity_processor, logger=None):
        super().__init__(logger)
        self.similarity_processor = similarity_processor
    
    def get_name(self) -> str:
        return "UEI"
    
    def compute(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute UEI for each external reason
        
        Args:
            sample_data: Contains external_reasons, external_reasons_confidences,
                        initial_reasons, and initial_reasons_confidences
            
        Returns:
            Dictionary with UEI scores
        """
        external_reasons = sample_data.get('external_reasons', [])
        external_confidences = sample_data.get('external_reasons_confidences', [])
        initial_reasons = sample_data.get('initial_reasons', [])
        initial_confidences = sample_data.get('initial_reasons_confidences', [])
        
        if not external_reasons:
            self.log_warning("No external reasons found for UEI computation")
            return {}
        
        uei_scores = {}
        for reason_ix in range(len(external_reasons)):
            confidence = external_confidences[reason_ix]
            
            # Compute between-runs diversity
            between_runs_diversity = self.similarity_processor.compute_between_runs_similarity(
                external_reasons[reason_ix],
                initial_reasons,
                initial_confidences,
                diversity=True
            )
            
            uei_score = (arc_hp.UII_Prediction_Weight * confidence) + \
                       (arc_hp.UII_Diversity_Weight * between_runs_diversity)
            
            uei_scores[f'reason_{reason_ix}'] = uei_score
        
        return uei_scores
