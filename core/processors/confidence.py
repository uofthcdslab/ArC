"""Confidence computation processor"""
import torch
import numpy as np
from typing import Tuple, List
from utils import helpers as hp


class ConfidenceProcessor:
    """Handles confidence computation logic"""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def compute_confidence(self, start_ix: int, out_tokens: List[int], 
                          reason_tokens: List[int], entropies: torch.Tensor,
                          relevances: List[float]) -> Tuple[float, bool]:
        """
        Compute confidence score for a reason
        
        Args:
            start_ix: Starting index in output tokens
            out_tokens: Output token IDs
            reason_tokens: Reason token IDs
            entropies: Token-wise entropies
            relevances: Token-wise relevance scores
            
        Returns:
            Tuple of (confidence_score, encoding_issue_flag)
        """
        if not out_tokens or not reason_tokens:
            return np.nan, False
        
        # Find common sublists
        reason_adj, out_adj, max_len = hp.get_common_sublists(reason_tokens, out_tokens)
        
        # Check for encoding issues
        encoding_issue = False
        if abs(len(reason_tokens) - max_len) > 4 or abs(len(out_tokens) - max_len) > 4:
            encoding_issue = True
        
        # Compute token-wise predictive entropies
        pe = entropies[(start_ix + out_adj):(start_ix + out_adj + max_len)].to('cpu')
        
        # Compute token-wise relevances
        rel = relevances[reason_adj:(reason_adj + max_len)]
        rel = [r / sum(rel) for r in rel]  # Length normalization
        
        # Token SAR (Sufficiency-Adjusted Relevance), generative probability
        token_sar = sum([p * r for p, r in zip(pe, rel)])
        confidence = torch.exp(-token_sar).item()
        
        return confidence, encoding_issue
    
    def get_indices(self, target_tokens: torch.Tensor, 
                   output_tokens: torch.Tensor) -> Tuple[int, int]:
        """
        Find indices of target tokens in output tokens
        
        Args:
            target_tokens: Target token IDs to find
            output_tokens: Output token IDs to search in
            
        Returns:
            Tuple of (start_index, end_index)
        """
        matching_indices = torch.nonzero(torch.isin(output_tokens, target_tokens), as_tuple=True)[0]
        
        # Handle case where no matches are found
        if len(matching_indices) == 0:
            if self.logger:
                self.logger.warning("No matches found for target tokens")
            return (0, 0)
        
        matching_indices_diff = torch.cat([torch.tensor([0]), torch.diff(matching_indices)])
        cont_matches = (matching_indices_diff == 1).int()
        cont_matches = torch.diff(torch.cat([torch.tensor([0]), cont_matches, torch.tensor([0])]))
        starts = (cont_matches == 1).nonzero(as_tuple=True)[0]
        ends = (cont_matches == -1).nonzero(as_tuple=True)[0]
        lengths = ends - starts
        max_idx = torch.argmax(lengths)
        
        return ((matching_indices[starts[max_idx]] - 1).item(), 
                (matching_indices[ends[max_idx] - 1] + 1).item())
