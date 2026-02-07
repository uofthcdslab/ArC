"""Individual reliance dimension metrics: RS and RN"""
import numpy as np
from typing import Dict, Any
from .base import BaseMetric


class RSMetric(BaseMetric):
    """Reason Sufficiency metric"""
    
    def __init__(self, similarity_processor, decision_importance_map, logger=None):
        super().__init__(logger)
        self.similarity_processor = similarity_processor
        self.decision_importance_map = decision_importance_map
    
    def get_name(self) -> str:
        return "RS"
    
    def compute(self, sample_data: Dict[str, Any]) -> Dict[int, float]:
        """
        Compute RS for each individual subsample
        
        Args:
            sample_data: Contains individual_reasons, individual_reasons_confidences,
                        individual_decision_confidence, new_decisions,
                        initial_reasons, and initial_reasons_confidences
            
        Returns:
            Dictionary with RS scores for each subsample
        """
        individual_reasons = sample_data.get('individual_reasons', [])
        individual_confidences = sample_data.get('individual_reasons_confidences', {})
        decision_confidences = sample_data.get('individual_decision_confidence', {})
        new_decisions = sample_data.get('new_decisions', [])
        initial_reasons = sample_data.get('initial_reasons', [])
        initial_confidences = sample_data.get('initial_reasons_confidences', [])
        
        rs_scores = {}
        
        for subsample_ix in range(len(individual_reasons)):
            # Part 1: Decision importance
            decision_imp = self.decision_importance_map.get(new_decisions[subsample_ix], 0.1)
            
            # Part 2: Decision confidence
            decision_conf = decision_confidences.get(subsample_ix, 0.0)
            
            # Part 3: Additional informativeness
            if len(individual_reasons[subsample_ix]) == 0:
                additional_informativeness = 0
            else:
                additional_informativeness = 0
                for reason_ix in range(len(individual_reasons[subsample_ix])):
                    confidence = individual_confidences[subsample_ix][reason_ix]
                    
                    # Target reasons exclude the current subsample
                    target_reasons = initial_reasons[:subsample_ix] + initial_reasons[subsample_ix+1:]
                    target_confidences = initial_confidences[:subsample_ix] + initial_confidences[subsample_ix+1:]
                    
                    between_runs_diversity = self.similarity_processor.compute_between_runs_similarity(
                        individual_reasons[subsample_ix][reason_ix],
                        target_reasons,
                        target_confidences,
                        diversity=True
                    )
                    
                    additional_informativeness += ((0.5 * confidence) + (0.5 * between_runs_diversity))
                
                additional_informativeness /= len(individual_reasons[subsample_ix])
            
            # For RS, we want 1 - additional_informativeness
            additional_informativeness = 1 - additional_informativeness
            
            # Final RS score
            rs_score = decision_imp * decision_conf * additional_informativeness
            rs_scores[subsample_ix] = rs_score
        
        return rs_scores


class RNMetric(BaseMetric):
    """Reason Necessity metric"""
    
    def __init__(self, similarity_processor, decision_importance_map, logger=None):
        super().__init__(logger)
        self.similarity_processor = similarity_processor
        self.decision_importance_map = decision_importance_map
    
    def get_name(self) -> str:
        return "RN"
    
    def compute(self, sample_data: Dict[str, Any]) -> Dict[int, float]:
        """
        Compute RN for each individual subsample
        
        Args:
            sample_data: Contains individual_reasons, individual_reasons_confidences,
                        individual_decision_confidence, new_decisions,
                        initial_reasons, and initial_reasons_confidences
            
        Returns:
            Dictionary with RN scores for each subsample
        """
        individual_reasons = sample_data.get('individual_reasons', [])
        individual_confidences = sample_data.get('individual_reasons_confidences', {})
        decision_confidences = sample_data.get('individual_decision_confidence', {})
        new_decisions = sample_data.get('new_decisions', [])
        initial_reasons = sample_data.get('initial_reasons', [])
        initial_confidences = sample_data.get('initial_reasons_confidences', [])
        
        rn_scores = {}
        
        for subsample_ix in range(len(individual_reasons)):
            # Part 1: Decision importance
            decision_imp = self.decision_importance_map.get(new_decisions[subsample_ix], 0.1)
            
            # Part 2: Decision confidence
            decision_conf = decision_confidences.get(subsample_ix, 0.0)
            
            # Part 3: Additional informativeness
            if len(individual_reasons[subsample_ix]) == 0:
                additional_informativeness = 0.01  # Penalty for no reasons
            else:
                additional_informativeness = 0
                for reason_ix in range(len(individual_reasons[subsample_ix])):
                    confidence = individual_confidences[subsample_ix][reason_ix]
                    
                    # Target similarity with the corresponding initial reason
                    target_similarity = float(self.similarity_processor.sims_hp.predict((
                        individual_reasons[subsample_ix][reason_ix],
                        initial_reasons[subsample_ix]
                    )))
                    target_similarity = target_similarity * initial_confidences[subsample_ix]
                    
                    additional_informativeness += ((0.5 * confidence) + (0.5 * target_similarity))
                
                additional_informativeness /= len(individual_reasons[subsample_ix])
            
            # Final RN score
            rn_score = decision_imp * decision_conf * additional_informativeness
            rn_scores[subsample_ix] = rn_score
        
        return rn_scores
