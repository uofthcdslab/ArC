"""Relevance dimension metrics: SoS and DiS"""
import numpy as np
from typing import Dict, Any
from .base import BaseMetric
from utils import arc_hyperparams as arc_hp
from utils import helpers as hp


class SoSMetric(BaseMetric):
    """Sufficiency of Stance metric"""
    
    def get_name(self) -> str:
        return "SoS"
    
    def compute(self, sample_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute SoS for each reason
        
        Args:
            sample_data: Contains initial_reasons_confidences and initial_reasons_sims_input
            
        Returns:
            Dictionary with SoS scores for each reason
        """
        confidences = sample_data.get('initial_reasons_confidences', [])
        sims_input = sample_data.get('initial_reasons_sims_input', [])
        
        if not confidences or not sims_input:
            self.log_warning("Missing data for SoS computation")
            return {}
        
        sos_scores = {}
        for reason_ix in range(len(confidences)):
            confidence = confidences[reason_ix]
            sim = sims_input[reason_ix]
            
            sos_score = (arc_hp.SoS_Prediction_Weight * confidence) + \
                       (arc_hp.SoS_Similarity_Weight * sim)
            
            sos_scores[f'reason_{reason_ix}'] = sos_score
        
        return sos_scores


class DiSMetric(BaseMetric):
    """Diversity of Stance metric"""
    
    def get_name(self) -> str:
        return "DiS"
    
    def compute(self, sample_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute DiS using average method
        
        Args:
            sample_data: Contains initial_reasons, initial_reasons_confidences, 
                        and initial_reasons_sims_reasons
            
        Returns:
            Dictionary with DiS_avg score
        """
        reasons = sample_data.get('initial_reasons', [])
        confidences = sample_data.get('initial_reasons_confidences', [])
        sims_reasons = sample_data.get('initial_reasons_sims_reasons', [])
        
        if len(reasons) <= 1:
            return {'DiS_avg': np.nan}
        
        # Create probability weights and similarity matrix
        prob_weights = hp.convert_list_to_col_matrix(confidences)
        similarity_matrix = hp.get_reasons_similarity_matrix(reasons, sims_reasons)
        
        if similarity_matrix.shape != prob_weights.shape:
            self.log_error(f"Shape mismatch: similarity_matrix {similarity_matrix.shape} vs prob_weights {prob_weights.shape}")
            return {'DiS_avg': np.nan}
        
        # Compute DiS-Avg
        tot_nas = 0
        dis_avg = hp.get_average_from_matrix((1 - similarity_matrix) * prob_weights, tot_nas=tot_nas)
        
        return {'DiS_avg': dis_avg}
