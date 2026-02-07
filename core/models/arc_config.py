"""ArC (Argument-based Consistency) Configuration dataclass"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ArCConfig:
    """Configuration for ArC computation"""
    
    explicit_prompting: str = '_explicit'
    use_scores: bool = False
    similarity_model: str = "cross-encoder/stsb-distilroberta-base"
    
    @property
    def entropy_mode(self) -> str:
        """Get entropy mode based on use_scores"""
        return 'scores' if self.use_scores else 'logits'
    
    @classmethod
    def from_args(cls, explicit_prompting: bool = True, 
                  use_scores: bool = False,
                  similarity_model: str = "cross-encoder/stsb-distilroberta-base"):
        """
        Create config from arguments
        
        Args:
            explicit_prompting: Whether to use explicit prompting
            use_scores: Whether to use scores instead of logits
            similarity_model: Name of similarity model
            
        Returns:
            ArCConfig instance
        """
        explicit_str = '_explicit' if explicit_prompting else ''
        return cls(
            explicit_prompting=explicit_str,
            use_scores=use_scores,
            similarity_model=similarity_model
        )
