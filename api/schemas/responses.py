"""Response schemas for ArC API"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class ComputeResponse(BaseModel):
    """Response for computation request"""
    success: bool
    message: str
    samples_computed: Optional[int] = None


class TaskStatus(BaseModel):
    """Status of a computation task"""
    task_id: str
    status: str  # "processing", "completed", "failed"
    progress: Optional[float] = None
    message: Optional[str] = None


class MetricSummary(BaseModel):
    """Summary statistics for a metric"""
    mean: float
    std: float
    min: float
    max: float
    count: int


class SummaryResponse(BaseModel):
    """Summary of ArC metrics for a model/dataset"""
    model_name: str
    data_name: str
    total_samples: int
    metrics: Dict[str, MetricSummary]


class SampleResultResponse(BaseModel):
    """Detailed results for a single sample"""
    sample_idx: int
    model_name: str
    data_name: str
    result: Dict[str, Any]


class ModelInfo(BaseModel):
    """Information about a model"""
    name: str
    short_name: str
    available_datasets: List[str]


class DatasetInfo(BaseModel):
    """Information about a dataset"""
    name: str
    available_models: List[str]
    sample_count: Optional[int] = None
