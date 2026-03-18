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
        Compute UII for each uphold_reasons_internal reason

        Args:
            sample_data: Contains uphold_reasons_internal_reasons, uphold_reasons_internal_reasons_confidences,
                        justify_reasons, and justify_reasons_confidences

        Returns:
            Dictionary with UII scores
        """
        uphold_reasons_internal_reasons = sample_data.get('uphold_reasons_internal_reasons', [])
        uphold_reasons_internal_confidences = sample_data.get('uphold_reasons_internal_reasons_confidences', [])
        justify_reasons = sample_data.get('justify_reasons', [])
        justify_confidences = sample_data.get('justify_reasons_confidences', [])

        if not uphold_reasons_internal_reasons:
            self.log_warning("No uphold_reasons_internal reasons found for UII computation")
            return {}

        uii_scores = {}
        for reason_ix in range(len(uphold_reasons_internal_reasons)):
            confidence = uphold_reasons_internal_confidences[reason_ix]

            # Compute between-runs diversity
            between_runs_diversity = self.similarity_processor.compute_between_runs_similarity(
                uphold_reasons_internal_reasons[reason_ix],
                justify_reasons,
                justify_confidences,
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
        Compute UEI for each uphold_reasons_external reason

        Args:
            sample_data: Contains uphold_reasons_external_reasons, uphold_reasons_external_reasons_confidences,
                        justify_reasons, and justify_reasons_confidences

        Returns:
            Dictionary with UEI scores
        """
        uphold_reasons_external_reasons = sample_data.get('uphold_reasons_external_reasons', [])
        uphold_reasons_external_confidences = sample_data.get('uphold_reasons_external_reasons_confidences', [])
        justify_reasons = sample_data.get('justify_reasons', [])
        justify_confidences = sample_data.get('justify_reasons_confidences', [])

        if not uphold_reasons_external_reasons:
            self.log_warning("No uphold_reasons_external reasons found for UEI computation")
            return {}

        uei_scores = {}
        for reason_ix in range(len(uphold_reasons_external_reasons)):
            confidence = uphold_reasons_external_confidences[reason_ix]

            # Compute between-runs diversity
            between_runs_diversity = self.similarity_processor.compute_between_runs_similarity(
                uphold_reasons_external_reasons[reason_ix],
                justify_reasons,
                justify_confidences,
                diversity=True
            )

            uei_score = (arc_hp.UII_Prediction_Weight * confidence) + \
                       (arc_hp.UII_Diversity_Weight * between_runs_diversity)

            uei_scores[f'reason_{reason_ix}'] = uei_score

        return uei_scores
