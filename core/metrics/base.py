"""Base metric class for ArC metrics"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseMetric(ABC):
    """Base class for all ArC metrics"""
    
    def __init__(self, logger=None):
        self.logger = logger
    
    @abstractmethod
    def compute(self, sample_data: Dict[str, Any]) -> Any:
        """
        Compute metric for a single sample
        
        Args:
            sample_data: Dictionary containing all necessary data for computation
            
        Returns:
            Computed metric value(s)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return metric name"""
        pass
    
    def log_info(self, message: str):
        """Log info message if logger is available"""
        if self.logger:
            self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message if logger is available"""
        if self.logger:
            self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message if logger is available"""
        if self.logger:
            self.logger.error(message)
